"""Frozen advanced experiments, real agent runs and original-page assistant reviews."""
import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import INSTRUCTIONS, VALIDATE_PROMPT, run_agent
from src.common import (DAY6, DAY6_CONFIG, DIMENSIONS, EMBEDDING_MODEL, MODEL, ROOT, api_client,
                        digest, read_json, record_usage, write_json)
from src.evaluation import evidence_complete
from src.review import review_case, summarize_reviews
from src.storage import Retriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--suite', choices=['advanced','basic_v2'], default='advanced')
    parser.add_argument('--experiment', default='main')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--top-k', type=int, default=6)
    parser.add_argument('--budget', type=int, default=6500)
    parser.add_argument('--max-rounds', type=int, default=6)
    parser.add_argument('--benchmark-version', type=int, default=1)
    parser.add_argument('--reuse-from', type=Path)
    args = parser.parse_args()
    check=api_client().responses.create(model=MODEL,input='Reply OK.',max_output_tokens=16)
    record_usage('benchmark_preflight',check.model,check.usage)
    corpus = 'wdr' if args.suite == 'advanced' else 'day6'
    version_suffix = '' if args.benchmark_version==1 else f'_v{args.benchmark_version}'
    cases_path = ROOT/f'benchmark/{args.suite}_cases{version_suffix}.json'
    cases = read_json(cases_path)
    freeze = read_json(ROOT/f'benchmark/{args.suite}_freeze{version_suffix}.json')
    if freeze['cases_sha256'] != digest(cases_path.read_bytes()):
        raise ValueError('Advanced questions changed after freeze')
    original = ROOT/'data/documents/WDR2024.pdf' if corpus=='wdr' else DAY6/'data/units.jsonl'
    if freeze.get('pdf_sha256',freeze.get('corpus_sha256')) != digest(original.read_bytes()):
        raise ValueError('Original document changed after freeze')
    retriever = Retriever(corpus)
    if corpus=='day6':
        from src.storage import bge_model
        import torch
        torch.set_num_threads(4)
        bge_model()
    selected = [case for case in cases if case['split'] == args.split]
    pipeline = {'model': MODEL, 'embedding_model': EMBEDDING_MODEL if corpus=='wdr' else DAY6_CONFIG['model'],
                'dimensions': DIMENSIONS if corpus=='wdr' else 768,
                'instruction_hash': digest(INSTRUCTIONS), 'chunks_hash': digest(retriever.chunks),
                'storage_code_hash': digest((ROOT/'src/storage.py').read_bytes()),
                'benchmark_hash': freeze['cases_sha256'], 'top_k': args.top_k,
                'budget': args.budget, 'max_rounds': args.max_rounds}
    if args.verify:
        pipeline.update(answer_verification=True, validation_instruction_hash=digest(VALIDATE_PROMPT))
    suffix = '' if args.experiment=='main' else '_' + args.experiment
    output = ROOT/f'results/{args.suite}{version_suffix}{suffix}'
    output.mkdir(parents=True, exist_ok=True)
    quota_exhausted=threading.Event()
    selection_path = output/'selection.json'
    if args.split == 'test':
        selection = read_json(selection_path)
        if selection.get('status') == 'incomplete_due_to_errors':
            raise ValueError('Complete dev reviews before selecting a test configuration')
        if selection['pipeline'] != pipeline:
            raise ValueError('Pipeline differs from the configuration locked on dev')
        methods = list(dict.fromkeys(['dense', selection['method']])) if args.suite=='basic_v2' else [selection['method']]
    else:
        methods = ['dense', 'bm25', 'hybrid']
        write_json(output/'selection_policy.json', {
            'priority': ['dev assistant-reviewed pass count', 'dev retrieval Evidence Hit@3', 'lower total agent time'],
            'human_confirmation': 'pending', 'pipeline': pipeline})

    def run_one(method, case):
        folder = output/args.split/method
        folder.mkdir(parents=True, exist_ok=True)
        trace_path = folder/f'{case["id"]}.json'
        if args.reuse_from and not trace_path.exists():
            original_path=args.reuse_from/args.split/method/f'{case["id"]}.json'
            if original_path.exists():
                source_selection=read_json(args.reuse_from/'selection.json')
                source_pipeline={k:v for k,v in source_selection['pipeline'].items() if k!='benchmark_hash'}
                current_pipeline={k:v for k,v in pipeline.items() if k!='benchmark_hash'}
                original_trace=read_json(original_path)
                if (source_pipeline==current_pipeline and original_trace.get('question')==case['question']
                        and original_trace.get('status')!='error'):
                    trace={**original_trace,'pipeline_hash':digest(pipeline),
                           'original_pipeline_hash':original_trace['pipeline_hash'],
                           'reused_from':str(original_path.relative_to(ROOT))}
                    write_json(trace_path,trace)
        if (args.resume or args.reuse_from) and trace_path.exists():
            trace = read_json(trace_path)
            if trace.get('status') != 'error':
                if trace.get('pipeline_hash') != digest(pipeline):
                    raise ValueError('Cannot resume with changed pipeline')
                return method, case, trace
        try:
            if quota_exhausted.is_set():
                blocked={'query_id':case['id'],'status':'error','error_type':'QuotaExhausted',
                         'pipeline_hash':digest(pipeline)}
                write_json(trace_path,blocked)
                return method,case,blocked
            ranking = retriever.search(case['question'], method, top_k=args.top_k)
            trace = run_agent(case['question'], retriever, method=method, top_k=args.top_k,
                              budget=args.budget, max_rounds=args.max_rounds, verify=args.verify)
            trace.update(query_id=case['id'], initial_ranking=ranking,
                         pipeline_hash=digest(pipeline))
            if case['answerable']:
                trace['retrieval_hit1'] = evidence_complete(case, ranking[:1])
                trace['retrieval_hit3'] = evidence_complete(case, ranking[:3])
                trace['context_evidence'] = evidence_complete(case, trace['context'])
        except Exception as error:
            body=getattr(error,'body',{}) or {}
            if body.get('code') in {'credit_balance_exhausted','insufficient_quota'}:
                quota_exhausted.set()
            trace = {'query_id': case['id'], 'status': 'error', 'error_type': type(error).__name__,
                     'pipeline_hash': digest(pipeline)}
        write_json(trace_path, trace)
        return method, case, trace

    runs = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_one, method, case) for method in methods for case in selected]
        for job in as_completed(jobs):
            method, case, trace = job.result()
            runs.append((method, case, trace))
            print(args.split, method, case['id'], trace['status'], flush=True)

    def review_one(run):
        method, case, trace = run
        path = output/args.split/method/f'{case["id"]}.review.json'
        if trace['status'] == 'error':
            review = {'query_id': case['id'], 'passed': False, 'review_error': trace['error_type'],
                      'human_confirmed': False}
        else:
            try:
                if quota_exhausted.is_set():
                    blocked={'query_id':case['id'],'passed':False,
                             'review_error':'QuotaExhausted','human_confirmed':False}
                    write_json(path,blocked)
                    return method,case,trace,blocked
                review = review_case(case, trace, corpus)
            except Exception as error:
                body=getattr(error,'body',{}) or {}
                if body.get('code') in {'credit_balance_exhausted','insufficient_quota'}:
                    quota_exhausted.set()
                review = {'query_id': case['id'], 'passed': False, 'review_error': type(error).__name__,
                          'human_confirmed': False}
        write_json(path, review)
        print('review', method, case['id'], review['passed'], flush=True)
        return method, case, trace, review

    reviews = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(review_one, sorted(runs, key=lambda row: (row[0], row[1]['id']))):
            reviews.append(result)
    summaries = []
    for method in methods:
        subset = [row for row in reviews if row[0] == method]
        answerable = [row for row in subset if row[1]['answerable']]
        summary = summarize_reviews(selected, [row[3] for row in subset])
        summary.update(method=method, split=args.split,
                       hit1={'n': sum(row[2].get('retrieval_hit1', False) for row in answerable), 'N': len(answerable)},
                       hit3={'n': sum(row[2].get('retrieval_hit3', False) for row in answerable), 'N': len(answerable)},
                       context_evidence={'n': sum(row[2].get('context_evidence', False) for row in answerable), 'N': len(answerable)},
                       agent_errors=sum(row[2]['status']=='error' for row in subset),
                       total_agent_seconds=sum(row[2].get('seconds',0) for row in subset))
        summaries.append(summary)
    write_json(output/args.split/'summary.json', summaries)
    if args.split == 'dev':
        if any(row['errors'] or row['agent_errors'] for row in summaries):
            write_json(output/'selection_pending.json', {'status':'incomplete_due_to_errors',
                       'pipeline':pipeline,'reason':'All dev runs and reviews must finish before ranking configurations.'})
            if selection_path.exists():
                existing=read_json(selection_path)
                existing['status']='incomplete_due_to_errors'
                write_json(selection_path,existing)
            print('Selection pending because some agent runs or reviews failed.',flush=True)
            return
        best = max(summaries, key=lambda row: (row['passed'], row['hit3']['n'], -row['total_agent_seconds']))
        write_json(selection_path, {'method': best['method'], 'pipeline': pipeline,
                   'status':'selected_on_complete_dev',
                   'selected_at': datetime.now(timezone.utc).isoformat(),
                   'rule': 'Dev assistant-reviewed pass count, then Evidence Hit@3, then lower total agent time.',
                   'human_confirmation': 'pending'})
    print(summaries, flush=True)


if __name__ == '__main__':
    main()
