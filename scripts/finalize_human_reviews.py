"""Import explicit student source confirmations and compute the final advanced score."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,digest,read_json,write_json
from src.review import summarize_reviews


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--file',type=Path,required=True)
    args=parser.parse_args()
    data=read_json(args.file)
    cases=read_json(ROOT/'benchmark/advanced_cases_v2.json')
    test=[case for case in cases if case['split']=='test']
    selection=read_json(ROOT/'results/final_selection.json')
    if data.get('split')!='test' or data.get('cases_hash')!=digest(cases):
        raise ValueError('Review file does not match the final advanced test')
    if data.get('pipeline_hash')!=digest(selection['pipeline']):
        raise ValueError('Review file does not match the selected pipeline')
    items=data.get('items',[])
    if len(items)!=len(test) or {row['query_id'] for row in items}!={case['id'] for case in test}:
        raise ValueError('Every test case must be reviewed exactly once')
    if not data.get('reviewer','').strip() or not data.get('reviewed_at'):
        raise ValueError('Reviewer identity and review time are required')
    if any(row.get('human_confirmed') is not True or not isinstance(row.get('passed'),bool)
           or not row.get('reason','').strip() for row in items):
        raise ValueError('Every case requires an explicit confirmation, boolean label and reason')
    summary=summarize_reviews(test,items)
    summary.update(human_confirmed=True,review_type='student_original_source_review',
                   reviewer=data['reviewer'],reviewed_at=data['reviewed_at'],
                   meets_90_percent=summary['passed']/summary['total']>=0.90)
    write_json(ROOT/'results/advanced_human_reviews.json',data)
    write_json(ROOT/'results/advanced_human_summary.json',summary)
    print(summary)


if __name__=='__main__':main()
