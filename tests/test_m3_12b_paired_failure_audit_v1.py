import importlib.util
from pathlib import Path
import unittest

P=Path(__file__).resolve().parents[1]/"scripts"/"m3_12b_paired_failure_audit_v1.py"
spec=importlib.util.spec_from_file_location("m312b",P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fixture(labels=(1,0), duplicate=False):
    a={"categories":{}}
    b={"categories":{}}
    for cat in m.CATS:
        count = 200 if cat in ("pcb1","pcb2") else 201
        images=[]
        for i in range(count):
            label = 1 if i < 100 else 0
            images.append({"image":f"{cat}/x/{i:03d}.JPG","label":label,"score":2.0 if i%2 else 0.0})
        a["categories"][cat]={"tiled":{"frozen_threshold":1.0,"per_image":images,
            "image_metrics":{"TP":50,"FN":50,"FP":(count-100)//2,"TN":(count-100)-(count-100)//2}}}
        b["categories"][cat]={"development_frozen_threshold":1.0,
            "per_image":[dict(row) for row in images],
            "image_metrics":{"TP":50,"FN":50,"FP":(count-100)//2,"TN":(count-100)-(count-100)//2}}
    return a,b

class TestAudit(unittest.TestCase):
    def test_cohort_outcomes(self):
        old,new=fixture()
        report,rows=m.outcomes(old,new)
        self.assertEqual(len(rows),802)
        self.assertEqual(report['totals']['defect_both_detect'],200)
        self.assertEqual(report['totals']['defect_both_miss'],200)
    def test_source_label_mismatch(self):
        old,new=fixture();new['categories']['pcb1']['per_image'][0]['label']=0
        with self.assertRaises(ValueError):m.outcomes(old,new)
    def test_duplicate(self):
        with self.assertRaises(ValueError):m.indexed([{'image':'x','label':1},{'image':'x','label':0}])
    def test_missing_image(self):
        old,new=fixture();new['categories']['pcb2']['per_image'].pop()
        with self.assertRaises(ValueError):m.outcomes(old,new)
    def test_confusion_mismatch(self):
        old,new=fixture();old['categories']['pcb4']['tiled']['image_metrics']['TP']=49
        with self.assertRaises(ValueError):m.outcomes(old,new)

if __name__=='__main__':unittest.main()
