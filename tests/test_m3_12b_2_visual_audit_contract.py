"""CPU-only tests for M3.12B-2 metadata, without opening VisA images."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "m3_12b_2_visual_failure_audit_v1.py"
spec = importlib.util.spec_from_file_location("m312b2", MODULE_PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class TestVisualAuditContract(unittest.TestCase):
    def test_fixed_case_count_and_cohorts(self):
        manifest = m.load_cases()
        self.assertEqual(len(manifest["cases"]), 24)
        self.assertEqual(len({r["image"] for r in manifest["cases"]}), 24)
        for cat in ["pcb1", "pcb2", "pcb3", "pcb4"]:
            rows = [r for r in manifest["cases"] if r["category"] == cat]
            self.assertEqual(tuple(r["outcome"] for r in rows), m.OUTCOMES)

    def test_expected_flags_consistent(self):
        for r in m.load_cases()["cases"]:
            self.assertEqual(r["m312_flag"], r["m312_score"] > r["m312_threshold"])
            self.assertEqual(r["tiled_flag"], r["tiled_score"] > r["tiled_threshold"])

    def test_test_label_groups_never_treated_as_calibration(self):
        self.assertEqual(len([r for r in m.load_cases()["cases"] if r["label"] == 1]), 16)
        self.assertEqual(len([r for r in m.load_cases()["cases"] if r["label"] == 0]), 8)

    def test_normal_mask_statistics(self):
        h = np.array([[0.1, 0.2],[0.3, 0.4]],dtype=np.float32)
        mask = np.zeros((2,2), dtype=bool)
        o = m.analysis_of_map(h,mask)
        self.assertEqual(o["ground_truth_area_fraction"],0)
        self.assertIsNone(o["fraction_of_top_1pct_heatmap_pixels_inside_gt"])
        self.assertIsNone(o["mean_heatmap_inside_gt"])

    def test_defect_mask_statistics(self):
        h = np.array([[0.1, 0.2],[0.3, 0.6]],dtype=np.float32)
        mask = np.array([[False,False],[False,True]])
        o = m.analysis_of_map(h,mask)
        self.assertEqual(o["ground_truth_area_fraction"],0.25)
        self.assertEqual(o["fraction_of_top_1pct_heatmap_pixels_inside_gt"],1)
        self.assertEqual(o["display_heatmap_fraction_above_fixed_vmax"],0.25)

    def test_immutable_json_refuses_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"x.json"
            m.write_immutable_json(path,{"sha":"old"})
            m.write_immutable_json(path,{"sha":"old"})
            with self.assertRaises(AssertionError): m.write_immutable_json(path,{"sha":"new"})
            self.assertEqual(json.loads(path.read_text()),{"sha":"old"})


if __name__ == "__main__": unittest.main()
