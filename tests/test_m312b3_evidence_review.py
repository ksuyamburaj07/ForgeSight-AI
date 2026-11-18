import csv
import io
import json
import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from m312b3_verify_evidence_v1 import verify, verify_rows, digest, EXPECTED

CSV_PATH=ROOT/'reports/eda/m312b3_provisional_reviewer_worksheet_v1.csv'
SUMMARY_PATH=ROOT/'reports/eda/m312b3_evidence_summary_v1.json'

class TestEvidenceDeskReview(unittest.TestCase):
    def read_rows(self):
        with CSV_PATH.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

    def test_24_original_cases(self):
        self.assertEqual(len(verify(CSV_PATH,SUMMARY_PATH)),24)

    def test_review_not_preapproved(self):
        rows=self.read_rows()
        self.assertEqual({r['review_status'] for r in rows},{'ASSISTANT_PRELIMINARY_HUMAN_PENDING'})
        self.assertTrue(all(not r['human_reviewer_observation'] for r in rows))

    def test_duplicate_rejected(self):
        rows=self.read_rows();rows[1]['image']=rows[0]['image']
        with self.assertRaises(AssertionError):verify_rows(rows)

    def test_changed_frozen_decision_rejected(self):
        rows=self.read_rows();rows[0]['m312_flag']='1'
        with self.assertRaises(AssertionError):verify_rows(rows)

    def test_human_review_cannot_overwrite_candidate(self):
        rows=self.read_rows(); rows[0]['human_reviewer_observation']='Reviewed'
        with self.assertRaises(AssertionError):verify_rows(rows)

    def test_editing_worksheet_breaks_hash(self):
        summary=json.loads(SUMMARY_PATH.read_text())
        self.assertEqual(summary['worksheet_sha256'],digest(CSV_PATH.read_bytes()))
        self.assertEqual(summary['input_report_sha256'],EXPECTED['report'])

if __name__=='__main__':unittest.main()
