"""Review saved basic traces against original Day 6 source units."""
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, read_json, write_json
from src.review import review_case, summarize_reviews


def main():
    cases = read_json(ROOT/'benchmark/basic_cases.json')
    by_id = {case['id']: case for case in cases}
    for split in ['dev', 'test']:
        rows = [row for row in read_json(ROOT/f'results/basic/{split}/traces.json') if 'agent' in row]

        def review(row):
            result = review_case(by_id[row['query_id']], row['agent'], 'day6')
            return {**result, 'method': row['method']}

        reviews = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(review, row) for row in rows]):
                result = future.result()
                reviews.append(result)
                write_json(ROOT/f'results/basic/{split}/assistant_reviews.json',
                           sorted(reviews, key=lambda row: (row['method'],row['query_id'])))
                print(split,result['method'],result['query_id'],result['passed'],flush=True)
        summaries = []
        for method in sorted({row['method'] for row in reviews}):
            subset = [row for row in reviews if row['method']==method]
            answerable = [row for row in subset if by_id[row['query_id']]['answerable']]
            missing = [row for row in subset if not by_id[row['query_id']]['answerable']]
            summary = summarize_reviews(cases,subset)
            summary.update(method=method,
                           context_grades={str(g):sum(row['context_grade']==g for row in answerable) for g in [0,1,2]},
                           answer_grades={str(g):sum(row['answer_grade']==g for row in answerable) for g in [0,1,2]},
                           grounded_answer_mean=sum(row['answer_grade'] for row in answerable)/len(answerable),
                           citation_support={'n':sum(sum(row['citation_support']) for row in subset),
                                             'N':sum(len(row['citation_support']) for row in subset)},
                           abstention={'n':sum(bool(row['abstention_correct']) for row in missing),'N':len(missing)})
            summaries.append(summary)
        write_json(ROOT/f'results/basic/{split}/assistant_review_summary.json',summaries)


if __name__=='__main__':
    main()
