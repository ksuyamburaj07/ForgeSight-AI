"""ForgeSight M3.12B-3: read-only integrity validator. stdlib only.

Never changes frozen benchmark, atlas files, thresholds, model weights, or review files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
from collections import Counter

EXPECTED = {
    'report': '0ab5d57553835f149ee22bdc6fde4b3323693e494beab698a335bf49ea923701',
    'selection': 'eb24292440134b06c155a25972fd32ca17c0fc0566b916715bc184b3b80b892b',
    'atlases': {
        'pcb1': 'c3c25cb5c9d1dee302d95fc9e2693678a1c7e16b73364518d34d17ce47aedf59',
        'pcb2': '204ed1d82d539382bfe9e0bf3c61c54b58ab36489609a0df8c72e38ecee380d1',
        'pcb3': 'f582ff36fe8a7d0ed0b4338ff81d842015a06bfeeacd2a72d83561851142076e',
        'pcb4': 'b9065b28a6a5229677c77538c241cfc496f744e74420aedce35b8c21b8c42fa7',
    },
}
OUTCOMES = (
    'defect_both_miss', 'defect_new_only', 'defect_tiled_only',
    'defect_both_detect', 'normal_new_only_false_alarm', 'normal_both_accepted',
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def verify_rows(rows: list[dict], record_lookup: dict | None = None) -> None:
    assert len(rows) == 24, 'Expected exactly 24 selected cases'
    assert len({r['image'] for r in rows}) == 24, 'Duplicate image in reviewer evidence'
    assert Counter(r['category'] for r in rows) == Counter({f'pcb{i}': 6 for i in range(1, 5)})
    assert Counter(r['frozen_outcome'] for r in rows) == Counter({outcome: 4 for outcome in OUTCOMES})
    for i, row in enumerate(rows, 1):
        assert row['case_id'] == f'B3-{i:02d}', 'Unexpected order/ID'
        assert row['preliminary_visual_observation'].strip()
        assert row['hypothesis_not_proven'].strip()
        assert row['remaining_uncertainty'].strip()
        assert row['review_status'] == 'ASSISTANT_PRELIMINARY_HUMAN_PENDING'
        for k in ('human_reviewer_observation','human_location_decision','human_evidence','human_uncertainty'):
            assert not row[k].strip(), 'Original candidate worksheet must remain pending human review'
        assert row['candidate_response_region'] in {
            'board_or_component', 'connector_or_pin', 'scene_outside_board',
            'scene_or_boundary_uncertain', 'mixed_or_uncertain',
        }
        assert int(row['label']) == int('/Anomaly/' in row['image'])
        assert bool(int(row['m312_flag'])) == (float(row['m312_score']) > float(row['m312_threshold']))
        if record_lookup:
            r = record_lookup[row['image']]
            assert r['outcome'] == row['frozen_outcome']
            assert r['category'] == row['category']
            assert r['label'] == int(row['label'])
            assert r['m312_flag'] == bool(int(row['m312_flag']))
            assert r['tiled_flag'] == bool(int(row['tiled_flag']))
            assert abs(r['m312_score'] - float(row['m312_score'])) < 1e-14
            assert abs(r['m312_threshold'] - float(row['m312_threshold'])) < 1e-14
            d = r['visual_diagnostics']
            assert abs(d['ground_truth_area_fraction'] - float(row['gt_area_fraction'])) < 1e-14
            if r['label']:
                assert abs(d['fraction_of_top_1pct_heatmap_pixels_inside_gt'] - float(row['fraction_top_1pct_inside_gt'])) < 1e-14
            else:
                assert row['fraction_top_1pct_inside_gt'] == ''


def verify_source_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as z:
        def get(filename: str) -> bytes:
            matches = [name for name in z.namelist() if name.endswith('/' + filename)]
            assert len(matches) == 1, f'Expected exactly one {filename} in source ZIP'
            return z.read(matches[0])
        report_bytes = get('m312b_2_visual_audit_report_v1.json')
        selection_bytes = get('m312b_2_fixed_case_selection_v1.json')
        assert digest(report_bytes) == EXPECTED['report'], 'B2 report fingerprint changed'
        assert digest(selection_bytes) == EXPECTED['selection'], 'B2 case-selection fingerprint changed'
        report, selection = json.loads(report_bytes), json.loads(selection_bytes)
        assert report['case_selection_sha256'] == EXPECTED['selection']
        assert len(report['records']) == 24 and len(selection['cases']) == 24
        assert [(r['image'],r['outcome']) for r in report['records']] == [(r['image'],r['outcome']) for r in selection['cases']]
        for category, sha in EXPECTED['atlases'].items():
            assert digest(get(f'm312b_2_{category}_visual_atlas_v1.png')) == sha
            assert report['visual_atlas_sha256'][category] == sha
        assert report['weights_updated'] is False and report['thresholds_modified'] is False
        return report


def verify(worksheet: Path, summary_path: Path, source_zip: Path | None = None):
    raw = worksheet.read_bytes()
    with io.StringIO(raw.decode('utf-8'), newline='') as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    assert summary['worksheet_sha256'] == digest(raw), 'Candidate CSV was modified'
    assert summary['input_report_sha256'] == EXPECTED['report']
    assert summary['input_case_selection_sha256'] == EXPECTED['selection']
    assert summary['model_weights_updated'] is False
    assert summary['thresholds_updated'] is False
    assert summary['benchmark_recomputed'] is False
    report = verify_source_zip(source_zip) if source_zip else None
    verify_rows(rows, {r['image']:r for r in report['records']} if report else None)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip',type=Path,help='The original 13MB M3.12B-2 Google Drive ZIP (optional for full source verification)')
    parser.add_argument('--worksheet',required=True,type=Path)
    parser.add_argument('--summary',required=True,type=Path)
    args=parser.parse_args()
    rows=verify(args.worksheet,args.summary,args.source_zip)
    print(f'M3.12B-3 READ-ONLY VERIFICATION PASS — {len(rows)} provisional cases')
    print('Source ZIP checked:', 'YES' if args.source_zip else 'NO — provide --source-zip for B2 atlas/report checks')
    print('No model or threshold updates. Human reviewer fields remain blank.')

if __name__=='__main__': main()
