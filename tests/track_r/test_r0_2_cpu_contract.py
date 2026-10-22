"""Contract tests for the R0.2 CPU manifest, no PAC path access."""
import importlib.util
from pathlib import Path

from prdiffusion.track_r.manifest import validate_row,uid_for

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_r02_expected_counts_and_exposed_sources_only():
    root=Path(__file__).resolve().parents[2]
    mod=load_module("r0_2_build_manifest",root/"scripts"/"track_r"/"r0_2_build_manifest.py")
    assert sum(mod.EXPECTED.values())==2520
    assert mod.EXPECTED[("S1_b22","DAPS")]==200
    assert mod.EXPECTED[("S2_b24_dev80","DAPS")]==320
    assert mod.EXPECTED[("S4_a15","FAST")]==32
    assert all("confirm" not in a.lower() for a,b in mod.EXPECTED)

def test_ffhq_source_derivation_is_explicit_not_an_original_gt_tensor():
    row={
        "uid":uid_for("S1_b22","DAPS","60044","fresh1"),
        "source":"S1_b22","family":"DAPS","image_id":"60044",
        "run_index":0,"run_label":"fresh1",
        "candidate_path":"/tmp/a.png","candidate_format":"png",
        "gt_path":"/tmp/ffhq_source.png","gt_format":"ffhq1024",
        "measurement_path":"/tmp/y.pt","measurement_sha256":None,
        "recorded_psnr_db":20.0,"recorded_psnr_convention":"B21_FLOAT_CLAMP",
        "native_selector_name":None,"native_selector_value":None,
        "recorded_cost_seconds":None,"recorded_unet_forwards":None,
        "notes":"source GT requires exact B21 transform"}
    assert validate_row(row)["gt_format"]=="ffhq1024"
