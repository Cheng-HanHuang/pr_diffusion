"""A8 registered forecast parser/interval contract; CPU-only synthetic."""
import unittest,tempfile,hashlib,json
from pathlib import Path
from unittest.mock import patch
from scripts.track_r import xp_a7_postfreeze_kernel_cpu as k

class RegisteredA8(unittest.TestCase):
    def test_realized_panel_interval_and_residual_floor(self):
        f={"point":.5,"lo95":.4999,"hi95":.5001,
           "interval_type":"predictive_realized_panel"}
        r=k.compare_interval(.50008,f)
        self.assertTrue(r["inside"])
        self.assertAlmostEqual(r["standardized_residual"],.00008/.005,places=8)
        self.assertFalse(k.compare_interval(.50011,f)["inside"])
    def test_registered_v2_and_backticked_commit(self):
        payload={"interval_type":"predictive_realized_panel",
                 "predictions":[{
                     "family":"FAST","cohort":"D1_fixed100","panel":"primary",
                     "panel_n_max":16,"model_variant":variant,"K":kk,
                     "policy":p,"point":.5,"lo95":.3,"hi95":.7
                 } for variant in ("race","S2","tail","race_tau1")
                   for kk in (8,12,16) for p in ("E0","O_raw","O_orbit")]}
        raw=json.dumps(payload).encode()
        digest=hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            registry=Path(temp)/"registry.md"
            quote=chr(96)
            registry.write_text(
                "| Freeze | Branch | Commit | Path | SHA-256 | UTC | Seen | Covers |\n"
                "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
                "| v2 | b | "+quote+"abc"+quote+" | "+quote+"x.json"+quote+
                " | "+quote+digest+quote+" | t | t | text |\n")
            def check(args,**kwargs):
                if args[-1]=="abc:x.json":return raw
                if args[-1]=="abc":return b"100\n"
                if args[-1]=="outcome":return b"200\n"
                raise RuntimeError("unexpected git args: "+repr(args))
            with patch.object(k.subprocess,"check_output",side_effect=check):
                _,rows=k.read_registered_forecasts(".",registry,
                     "v2","abc","x.json",digest,"outcome")
            self.assertEqual(len(rows),36)
            self.assertIn(("FAST","D1_fixed100","primary",16,"race",8,"E0"),rows)
    def test_unknown_scope_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError,"WRONG_SCOPE"):
            k.verify_freeze(".",Path("/nonexistent"),"v2 (optional)",
                "aaa","x","sha","bbb")
if __name__=="__main__":unittest.main()
