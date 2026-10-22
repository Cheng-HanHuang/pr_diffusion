"""CPU-only unit coverage for the R0.0b missing-dependency safety parser."""
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "track_r" / "r0_0b_reconcile_pip.py"
    spec = importlib.util.spec_from_file_location("r0b_reconcile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_dependency_is_classified_but_incompatibility_is_not():
    mod=_module()
    assert mod.missing_spec(
        "ftfy 6.3.1 requires wcwidth, which is not installed."
    ) == {"req": "ftfy", "version": "6.3.1", "dep": "wcwidth"}
    assert mod.missing_spec(
        "scikit-learn 1.7.0 requires threadpoolctl, which is not installed."
    )["dep"] == "threadpoolctl"
    assert mod.missing_spec(
        "scikit-learn 1.7.0 has requirement joblib>=1.2, but you have joblib 0.8."
    ) is None
    assert mod.missing_spec("No broken requirements found.") is None


def test_numerical_pins_are_protected():
    mod=_module()
    assert "torch" in mod.PROHIBITED
    assert "numpy" in mod.PROHIBITED
    assert "scipy" in mod.PROHIBITED
    assert "torchvision" in mod.PROHIBITED
