"""Check automatic concurrency leaves the specified GPU memory reserve."""

from choose_future_advantage_replay_shards import choose


def resource(peak):
    return {"schema": "future_advantage_sparse_smoke_resource_v1",
            "wall_seconds": 12, "samples": 10,
            "baseline_used_mib": 2000, "peak_used_mib": peak,
            "gpu_total_mib": 80000}


assert choose(resource(6000)) == 4
assert choose(resource(18000)) == 2
assert choose(resource(36000)) == 1
try:
    choose(resource(59000))
except ValueError:
    pass
else:
    raise AssertionError("accepted insufficient memory reserve")
print("sparse replay shard resource decision passed")
