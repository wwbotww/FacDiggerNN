from facdigger.experiments import manifest as manifest_module


def test_config_hash_is_order_independent() -> None:
    left = manifest_module.sha256_json({"b": 2, "a": 1})
    right = manifest_module.sha256_json({"a": 1, "b": 2})
    assert left == right
