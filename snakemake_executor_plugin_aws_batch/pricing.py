"""Instance price lookup for per-job cost estimation, with caching.

Cost estimation needs a per-hour price for the EC2 instance a job ran on. Two
sources, because they are fundamentally different:

- **On-demand list price** comes from the AWS Price List API (``pricing:GetProducts``).
  It is effectively static, so it is cached aggressively — in memory for the run
  and persisted to disk with a long TTL, so the steady state makes zero pricing
  calls.
- **Spot price** is dynamic, so it comes from ``ec2:DescribeSpotPriceHistory`` and
  is cached only in memory for the run (never persisted — yesterday's spot price
  is meaningless tomorrow).

Everything is best-effort: any AWS/parse error returns None, and the caller
simply omits the cost estimate rather than reporting a wrong number. The Price
List API is only available in us-east-1 / ap-south-1, so the pricing client is
created there regardless of the workflow's region (the region is a query filter).
"""

import json
import os
import time
from typing import Any, Callable, Optional

# Default persistent-cache TTL for on-demand list prices (they change rarely).
DEFAULT_TTL_SECONDS = 14 * 24 * 3600


def default_cache_path() -> str:
    """Path to the persistent on-demand price cache under the XDG cache dir."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "snakemake-executor-plugin-aws-batch", "pricing.json")


class PriceCache:
    """Two-layer price cache: in-memory always, optional persistent disk (TTL'd)."""

    def __init__(
        self,
        path: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.ttl = ttl_seconds
        self._now = now_fn
        self._mem: dict = {}
        self._disk: dict = self._load()

    def _load(self) -> dict:
        if not self.path or not os.path.exists(self.path):
            return {}
        try:
            with open(self.path) as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            # Re-read and merge before writing so two concurrent runs sharing the
            # cache file don't clobber each other's newly-added entries (our keys
            # win on conflict, which is fine — same price). Write atomically.
            merged = self._load()
            merged.update(self._disk)
            self._disk = merged
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w") as fh:
                json.dump(self._disk, fh)
            os.replace(tmp, self.path)
        except Exception:
            pass  # best-effort; a cache write failure must never break a run

    def get(self, key: str) -> Optional[float]:
        if key in self._mem:
            return self._mem[key]
        entry = self._disk.get(key)
        if not isinstance(entry, dict):
            return None
        price = entry.get("price")
        if price is not None and (self._now() - entry.get("ts", 0)) <= self.ttl:
            self._mem[key] = price
            return price
        return None

    def put(self, key: str, price: float, persist: bool = True) -> None:
        self._mem[key] = price
        if persist and self.path:
            self._disk[key] = {"price": price, "ts": self._now()}
            self._save()


def parse_ondemand_price(price_list_item: str) -> Optional[float]:
    """Extract the on-demand USD/hour from a Price List API product JSON string."""
    try:
        product = json.loads(price_list_item)
        on_demand = product["terms"]["OnDemand"]
        offer = next(iter(on_demand.values()))
        dimension = next(iter(offer["priceDimensions"].values()))
        usd = dimension["pricePerUnit"]["USD"]
        price = float(usd)
        return price if price > 0 else None
    except (KeyError, StopIteration, ValueError, TypeError):
        return None


def ondemand_price_per_hour(
    pricing_client: Any, instance_type: str, region: str
) -> Optional[float]:
    """Query the Price List API for a Linux/shared on-demand instance price."""
    try:
        response = pricing_client.get_products(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceType", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
        )
        for item in response.get("PriceList", []):
            price = parse_ondemand_price(item)
            if price is not None:
                return price
        return None
    except Exception:
        return None


def spot_price_per_hour(
    ec2_client: Any, instance_type: str, availability_zone: Optional[str]
) -> Optional[float]:
    """Get the most recent spot price for an instance type (best-effort)."""
    try:
        kwargs: dict = {
            "InstanceTypes": [instance_type],
            "ProductDescriptions": ["Linux/UNIX"],
            "MaxResults": 1,
        }
        if availability_zone:
            kwargs["AvailabilityZone"] = availability_zone
        response = ec2_client.describe_spot_price_history(**kwargs)
        history = response.get("SpotPriceHistory", [])
        if not history:
            return None
        price = float(history[0]["SpotPrice"])
        return price if price > 0 else None
    except Exception:
        return None


def compute_cost(
    price_per_hour_usd: Optional[float],
    started_at: Optional[float],
    stopped_at: Optional[float],
    job_vcpus: Optional[float] = None,
    instance_vcpus: Optional[float] = None,
) -> Optional[float]:
    """Estimate a job's cost in USD, apportioned by its share of the instance.

    cost = price/hour x hours x (job_vCPU / instance_vCPU). The vCPU share
    apportions a shared instance's cost across the jobs packed onto it; it
    defaults to 1.0 (whole instance) when the vCPU counts aren't known. Note that
    default over-counts a job that actually shared its instance with others (each
    job would be charged the full instance) — so the estimate is an upper bound
    when instance vCPUs can't be resolved. Returns None when price/window missing.
    """
    if price_per_hour_usd is None or started_at is None or stopped_at is None:
        return None
    hours = (stopped_at - started_at) / 3600.0
    if hours <= 0:
        return None
    share = 1.0
    if job_vcpus and instance_vcpus and instance_vcpus > 0:
        share = min(1.0, job_vcpus / instance_vcpus)
    return round(price_per_hour_usd * hours * share, 4)


def price_per_hour(
    instance_type: str,
    region: str,
    availability_zone: Optional[str],
    lifecycle: Optional[str],
    *,
    pricing_client: Any,
    ec2_client: Any,
    cache: PriceCache,
) -> Optional[float]:
    """Return the per-hour USD price for an instance, using the right source.

    Spot instances (``lifecycle == "spot"``) are priced from the spot market and
    cached in memory only; everything else uses the on-demand list price, cached
    persistently (list prices barely change).
    """
    is_spot = lifecycle == "spot"
    # On-demand list price is region-level (AZ-independent), so the AZ is only part
    # of the key for spot — keeping it out of the on-demand key avoids redundant
    # Price List calls for the same instance type across AZs.
    if is_spot:
        key = f"spot:{region}:{availability_zone or '-'}:{instance_type}"
    else:
        key = f"ondemand:{region}:{instance_type}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    if is_spot:
        price = spot_price_per_hour(ec2_client, instance_type, availability_zone)
        if price is not None:
            cache.put(key, price, persist=False)  # spot prices are not persisted
        return price

    price = ondemand_price_per_hour(pricing_client, instance_type, region)
    if price is not None:
        cache.put(key, price, persist=True)
    return price
