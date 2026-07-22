from facdigger.environment import environment_is_healthy


def test_environment_health_can_distinguish_core_and_model_dependencies() -> None:
    report = {
        "dependencies": [
            {"name": "pydantic", "importable": True},
            {"name": "PyYAML", "importable": True},
            {"name": "typer", "importable": True},
            {"name": "numpy", "importable": True},
            {"name": "torch", "importable": False},
            {"name": "transformers", "importable": False},
            {"name": "huggingface-hub", "importable": False},
        ]
    }

    assert environment_is_healthy(report, require_model=False)
    assert not environment_is_healthy(report, require_model=True)
