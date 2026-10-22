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


def test_exact_torch_setuptools_conflict_is_safely_pinned(monkeypatch):
    mod=_module()
    class FakeDist:
        requires=["setuptools<82"]
    versions={"torch":"2.11.0+cu128","setuptools":"82.0.1"}
    monkeypatch.setattr(mod.metadata,"version",lambda name: versions[name])
    monkeypatch.setattr(mod.metadata,"distribution",lambda name: FakeDist())
    issue="torch 2.11.0+cu128 has requirement setuptools<82, but you have setuptools 82.0.1."
    assert mod.pinned_setuptools_for(issue) == "setuptools==80.9.0"
    assert mod.pinned_setuptools_for(issue.replace("torch 2.11.0", "numpy 2.11.0")) is None
    assert mod.pinned_setuptools_for(issue.replace("setuptools<82", "setuptools<80")) is None
    assert mod.pinned_setuptools_for(issue.replace("82.0.1", "81.0.0")) is None


def test_core_clone_versions_guard(monkeypatch):
    mod=_module()
    base={name:"v1" for name in mod.CORE_PACKAGES}
    def matching(py, _names):
        return dict(base)
    monkeypatch.setattr(mod,"installed_versions",matching)
    assert mod.core_versions_match("source","target")
    def mismatching(py, _names):
        d=dict(base)
        if py=="target": d["torch"]="v2"
        return d
    monkeypatch.setattr(mod,"installed_versions",mismatching)
    assert not mod.core_versions_match("source","target")
