"""Tests for the instance price lookup and cache (mocked AWS, no creds)."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from snakemake_executor_plugin_aws_batch import Executor, pricing
from snakemake_executor_plugin_aws_batch import _container_vcpus, _ms_to_seconds


def _executor(**settings):
    base = {"region": "us-east-1", "estimate_cost": True}
    base.update(settings)
    ex = Executor.__new__(Executor)
    ex.logger = MagicMock()
    ex.settings = SimpleNamespace(**base)
    return ex


CI_ARN = "arn:aws:ecs:us-east-1:1:container-instance/c/abc"

# A trimmed Price List API product JSON (on-demand offer).
_PRODUCT = json.dumps(
    {
        "terms": {
            "OnDemand": {
                "ABC.JRTCKXETXF": {
                    "priceDimensions": {
                        "ABC.JRTCKXETXF.6YS6EN2CT7": {
                            "pricePerUnit": {"USD": "0.1920000000"}
                        }
                    }
                }
            }
        }
    }
)


class TestParseOndemandPrice:
    def test_parses_usd_per_hour(self):
        assert pricing.parse_ondemand_price(_PRODUCT) == 0.192

    def test_zero_price_is_none(self):
        item = _PRODUCT.replace("0.1920000000", "0.0000000000")
        assert pricing.parse_ondemand_price(item) is None

    def test_malformed_is_none(self):
        assert pricing.parse_ondemand_price("{}") is None
        assert pricing.parse_ondemand_price("not json") is None


class TestPriceCache:
    def test_memory_get_put(self):
        c = pricing.PriceCache(path=None)
        assert c.get("k") is None
        c.put("k", 1.5)
        assert c.get("k") == 1.5

    def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "p.json")
        c = pricing.PriceCache(path=path)
        c.put("k", 2.0)
        # A fresh cache reading the same file sees the persisted price.
        c2 = pricing.PriceCache(path=path)
        assert c2.get("k") == 2.0

    def test_ttl_expiry(self, tmp_path):
        path = str(tmp_path / "p.json")
        clock = [1000.0]
        c = pricing.PriceCache(path=path, ttl_seconds=100, now_fn=lambda: clock[0])
        c.put("k", 3.0)
        clock[0] = 1050.0  # within TTL
        assert (
            pricing.PriceCache(path=path, ttl_seconds=100, now_fn=lambda: clock[0]).get(
                "k"
            )
            == 3.0
        )
        clock[0] = 2000.0  # past TTL
        assert (
            pricing.PriceCache(path=path, ttl_seconds=100, now_fn=lambda: clock[0]).get(
                "k"
            )
            is None
        )

    def test_non_persisted_put_not_on_disk(self, tmp_path):
        path = str(tmp_path / "p.json")
        c = pricing.PriceCache(path=path)
        c.put("spot", 0.5, persist=False)
        assert c.get("spot") == 0.5  # in memory
        assert pricing.PriceCache(path=path).get("spot") is None  # not on disk

    def test_corrupt_cache_file_ignored(self, tmp_path):
        path = str(tmp_path / "p.json")
        with open(path, "w") as fh:
            fh.write("{ not valid json")
        c = pricing.PriceCache(path=path)
        assert c.get("anything") is None  # no crash

    def test_malformed_entry_without_price_is_none(self, tmp_path):
        path = str(tmp_path / "p.json")
        with open(path, "w") as fh:
            json.dump({"k": {"ts": 123}}, fh)  # entry missing "price"
        c = pricing.PriceCache(path=path)
        assert c.get("k") is None  # no KeyError

    def test_save_to_unwritable_path_is_noop(self):
        # A cache whose dir can't be created must not raise on put.
        c = pricing.PriceCache(path="/proc/nonexistent/dir/p.json")
        c.put("k", 1.0)  # must not raise
        assert c.get("k") == 1.0  # still in memory


class TestComputeCost:
    def test_full_instance(self):
        # 0.192/hr for 0.5h, whole instance.
        assert pricing.compute_cost(0.192, 1000.0, 2800.0) == 0.096

    def test_apportioned_by_vcpu_share(self):
        # 2 of 8 vCPUs -> a quarter of the instance cost.
        assert (
            pricing.compute_cost(0.4, 0.0, 3600.0, job_vcpus=2, instance_vcpus=8) == 0.1
        )

    def test_share_capped_at_one(self):
        assert (
            pricing.compute_cost(0.4, 0.0, 3600.0, job_vcpus=16, instance_vcpus=8)
            == 0.4
        )

    def test_none_inputs(self):
        assert pricing.compute_cost(None, 0.0, 3600.0) is None
        assert pricing.compute_cost(0.4, None, 3600.0) is None
        assert pricing.compute_cost(0.4, 100.0, 100.0) is None  # zero duration

    def test_negative_duration_is_none(self):
        assert pricing.compute_cost(0.4, 3600.0, 0.0) is None  # stopped before started

    def test_unknown_instance_vcpus_charges_full_instance(self):
        # The documented over-count: without instance vCPUs the share is 1.0.
        assert (
            pricing.compute_cost(0.4, 0.0, 3600.0, job_vcpus=2, instance_vcpus=None)
            == 0.4
        )


class TestOndemandPricePerHour:
    def test_returns_parsed_price(self):
        client = MagicMock()
        client.get_products.return_value = {"PriceList": [_PRODUCT]}
        assert pricing.ondemand_price_per_hour(client, "c5.large", "us-east-1") == 0.192

    def test_empty_pricelist_is_none(self):
        client = MagicMock()
        client.get_products.return_value = {"PriceList": []}
        assert pricing.ondemand_price_per_hour(client, "c5.large", "us-east-1") is None

    def test_api_error_is_none(self):
        client = MagicMock()
        client.get_products.side_effect = Exception("AccessDenied")
        assert pricing.ondemand_price_per_hour(client, "c5.large", "us-east-1") is None


class TestSpotPricePerHour:
    def test_returns_latest_spot_price(self):
        client = MagicMock()
        client.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [{"SpotPrice": "0.0345"}]
        }
        assert pricing.spot_price_per_hour(client, "c5.large", "us-east-1a") == 0.0345

    def test_no_history_is_none(self):
        client = MagicMock()
        client.describe_spot_price_history.return_value = {"SpotPriceHistory": []}
        assert pricing.spot_price_per_hour(client, "c5.large", None) is None


class TestPricePerHourRouting:
    def test_ondemand_uses_pricing_api_and_persists(self, tmp_path):
        cache = pricing.PriceCache(path=str(tmp_path / "p.json"))
        pricing_client = MagicMock()
        pricing_client.get_products.return_value = {"PriceList": [_PRODUCT]}
        ec2 = MagicMock()
        price = pricing.price_per_hour(
            "c5.large",
            "us-east-1",
            "us-east-1a",
            "on-demand",
            pricing_client=pricing_client,
            ec2_client=ec2,
            cache=cache,
        )
        assert price == 0.192
        ec2.describe_spot_price_history.assert_not_called()
        # Cached: a second call doesn't re-hit the pricing API.
        pricing.price_per_hour(
            "c5.large",
            "us-east-1",
            "us-east-1a",
            "on-demand",
            pricing_client=pricing_client,
            ec2_client=ec2,
            cache=cache,
        )
        assert pricing_client.get_products.call_count == 1

    def test_ondemand_key_ignores_az(self, tmp_path):
        # Same instance type in two AZs must reuse one on-demand price (one API call).
        cache = pricing.PriceCache(path=str(tmp_path / "p.json"))
        pricing_client = MagicMock()
        pricing_client.get_products.return_value = {"PriceList": [_PRODUCT]}
        ec2 = MagicMock()
        for az in ("us-east-1a", "us-east-1b"):
            pricing.price_per_hour(
                "c5.large",
                "us-east-1",
                az,
                "on-demand",
                pricing_client=pricing_client,
                ec2_client=ec2,
                cache=cache,
            )
        assert pricing_client.get_products.call_count == 1

    def test_spot_uses_ec2_history_and_does_not_persist(self, tmp_path):
        path = str(tmp_path / "p.json")
        cache = pricing.PriceCache(path=path)
        ec2 = MagicMock()
        ec2.describe_spot_price_history.return_value = {
            "SpotPriceHistory": [{"SpotPrice": "0.03"}]
        }
        pricing_client = MagicMock()
        price = pricing.price_per_hour(
            "c5.large",
            "us-east-1",
            "us-east-1a",
            "spot",
            pricing_client=pricing_client,
            ec2_client=ec2,
            cache=cache,
        )
        assert price == 0.03
        pricing_client.get_products.assert_not_called()
        # Spot price is not persisted to disk.
        assert (
            not os.path.exists(path)
            or pricing.PriceCache(path=path).get("spot:us-east-1:us-east-1a:c5.large")
            is None
        )


class TestModuleHelpers:
    def test_ms_to_seconds(self):
        assert _ms_to_seconds(142000) == 142.0
        assert _ms_to_seconds(None) is None
        assert _ms_to_seconds("bad") is None

    def test_container_vcpus_from_resource_requirements(self):
        container = {
            "resourceRequirements": [
                {"type": "VCPU", "value": "4"},
                {"type": "MEMORY", "value": "8192"},
            ]
        }
        assert _container_vcpus(container) == 4.0

    def test_container_vcpus_legacy_field(self):
        assert _container_vcpus({"vcpus": 2}) == 2.0

    def test_container_vcpus_absent(self):
        assert _container_vcpus({}) is None


class TestResolveInstanceDetails:
    def _ex(self):
        ex = _executor()
        ex._aws_clients = {
            "ecs": MagicMock(
                describe_container_instances=MagicMock(
                    return_value={"containerInstances": [{"ec2InstanceId": "i-abc"}]}
                )
            ),
            "ec2": MagicMock(
                describe_instances=MagicMock(
                    return_value={
                        "Reservations": [
                            {
                                "Instances": [
                                    {
                                        "InstanceType": "c5.2xlarge",
                                        "CpuOptions": {
                                            "CoreCount": 4,
                                            "ThreadsPerCore": 2,
                                        },
                                        "InstanceLifecycle": "spot",
                                        "Placement": {"AvailabilityZone": "us-east-1a"},
                                    }
                                ]
                            }
                        ]
                    }
                )
            ),
        }
        return ex

    def test_resolves_type_vcpus_lifecycle_az(self):
        ex = self._ex()
        details = ex._resolve_instance_details(
            {"container": {"containerInstanceArn": CI_ARN}}
        )
        assert details["instance_type"] == "c5.2xlarge"
        assert details["vcpus"] == 8  # 4 cores x 2 threads
        assert details["lifecycle"] == "spot"
        assert details["az"] == "us-east-1a"

    def test_no_instance_returns_none(self):
        ex = _executor()
        ex._aws_clients = {
            "ecs": MagicMock(
                describe_container_instances=MagicMock(
                    return_value={"containerInstances": []}
                )
            )
        }
        assert (
            ex._resolve_instance_details(
                {"container": {"containerInstanceArn": CI_ARN}}
            )
            is None
        )


class TestEstimateCost:
    def test_disabled_returns_none(self):
        ex = _executor(estimate_cost=False)
        assert ex._estimate_cost({"container": {}}) is None

    def test_end_to_end_ondemand(self, tmp_path):
        ex = _executor()
        ex._aws_clients = {
            "ecs": MagicMock(
                describe_container_instances=MagicMock(
                    return_value={"containerInstances": [{"ec2InstanceId": "i-abc"}]}
                )
            ),
            "ec2": MagicMock(
                describe_instances=MagicMock(
                    return_value={
                        "Reservations": [
                            {
                                "Instances": [
                                    {
                                        "InstanceType": "c5.large",
                                        "CpuOptions": {
                                            "CoreCount": 1,
                                            "ThreadsPerCore": 2,
                                        },
                                        "Placement": {"AvailabilityZone": "us-east-1a"},
                                    }
                                ]
                            }
                        ]
                    }
                )
            ),
            "pricing": MagicMock(
                get_products=MagicMock(return_value={"PriceList": [_PRODUCT]})
            ),
        }
        ex._price_cache_obj = pricing.PriceCache(path=None)
        job_info = {
            "container": {
                "containerInstanceArn": CI_ARN,
                "resourceRequirements": [{"type": "VCPU", "value": "2"}],
            },
            "startedAt": 0,
            "stoppedAt": 3600000,  # 1 hour
        }
        # price 0.192/hr, 1h, whole instance (2 of 2 vcpus) -> 0.192
        assert ex._estimate_cost(job_info) == 0.192

    def test_degrades_when_instance_unresolvable(self):
        ex = _executor()
        ex._aws_clients = {
            "ecs": MagicMock(
                describe_container_instances=MagicMock(
                    return_value={"containerInstances": []}
                )
            )
        }
        assert (
            ex._estimate_cost(
                {
                    "container": {"containerInstanceArn": CI_ARN},
                    "startedAt": 0,
                    "stoppedAt": 3600000,
                }
            )
            is None
        )
