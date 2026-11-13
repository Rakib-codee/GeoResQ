"""End-to-end API & UI smoke tests for the GeoResQ Interactive Visual Dashboard."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from georesq.dashboard.server import create_server


class TestGeoResQDashboard(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = create_server(host="127.0.0.1", port=0, repo_root=REPO_ROOT)
        cls.port = cls.server.server_address[1]
        cls.base_url = f"http://127.0.0.1:{cls.port}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3.0)

    def test_index_status_and_benchmarks_endpoints(self) -> None:
        with urlopen(f"{self.base_url}/") as r:
            html = r.read().decode("utf-8")
            self.assertEqual(r.status, 200)
            self.assertIn("GeoResQ v0.2", html)

        with urlopen(f"{self.base_url}/api/status") as r:
            st = json.loads(r.read().decode("utf-8"))
            self.assertEqual(len(st["builtin_scenes"]), 3)

        with urlopen(f"{self.base_url}/api/benchmarks") as r:
            bm = json.loads(r.read().decode("utf-8"))
            self.assertIn("stress_test_v0_2", bm)
            self.assertIn("failure_analysis_v0_2", bm)
            self.assertEqual(len(bm["per_tile_audit"]), 144)

    def test_predict_clean_and_corrupted_scenes(self) -> None:
        for corr, sev in [("clean", 0.0), ("cloud_occlusion", 0.75)]:
            req = Request(
                f"{self.base_url}/api/predict",
                data=json.dumps(
                    {
                        "scene_id": "builtin:port_blast",
                        "variant_id": "lite_gated_fusion_v0_2",
                        "corruption": corr,
                        "severity": sev,
                        "grid_size": 4,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req) as r:
                res = json.loads(r.read().decode("utf-8"))
            self.assertEqual(res["status"], "OK")
            self.assertTrue(res["human_review_required"])
            self.assertEqual(len(res["sectors"]), 16)
            self.assertTrue(res["layers"]["predicted_damage"].startswith("data:image/png;base64,"))
            self.assertTrue(res["layers"]["entropy"].startswith("data:image/png;base64,"))
            if sev > 0:
                self.assertGreater(res["corruption_diagnostics"]["opt_mae_u8"], 10.0)


if __name__ == "__main__":
    unittest.main()
