"""Comparable retrieval metrics and real agent traces; human labels stay separate."""
import time
import re
import inspect

from .agent import run_agent
from .common import MODEL, PROVIDER, ROOT, digest, read_json, write_json
from .local_agent import run_local_agent


def evaluation_fingerprint(cases,retriever):
    return digest([cases,retriever.chunks,PROVIDER,MODEL,digest(inspect.getsource(run_local_agent))])


def supported_gold(case, row):
    alternatives = case.get('gold') or [
        {'source_id': case.get('gold_source_id'), 'page': case.get('gold_page'),
         'evidence': case.get('evidence_quote', '')}]
    for gold in alternatives:
        quote = ' '.join(gold.get('evidence', gold.get('quote', '')).split())
        if (quote and row['source_id'] == gold['source_id'] and str(row['page']) == str(gold['page'])
                and quote in ' '.join(row['text'].split())):
            return True
    return False


def evidence_complete(case, rows):
    groups = case.get('evidence_groups')
    if not groups:
        return any(supported_gold(case, row) for row in rows)

    def matches(reference, row):
        if row['source_id'] != reference['source_id'] or str(row['page']) != str(reference['page']):
            return False
        quote = reference.get('evidence')
        if quote:
            return ' '.join(quote.split()) in ' '.join(row['text'].split())
        label = reference.get('region_label', '')
        if label.lower().startswith(('figure ', 'table ', 'map ')):
            normalized = re.sub(r'\s+', ' ', row.get('metadata', {}).get('label', '') + ' ' + row['text']).lower()
            return label.lower() in normalized
        return True

    return all(any(matches(reference, row) for reference in alternatives for row in rows)
               for alternatives in groups)


def evaluate_cases(cases, retriever, split, output, agent=False, agent_method='hybrid', resume=False):
    selected = [case for case in cases if case['split'] == split]
    if not selected:
        raise ValueError('No cases for selected split')
    fingerprint = evaluation_fingerprint(cases,retriever)
    if split == 'test':
        selection = read_json(output.parent / 'selection.json')
        if selection['fingerprint'] != fingerprint or selection['method'] != agent_method:
            raise ValueError('Freeze matching corpus, benchmark and method on dev before test')
    output.mkdir(parents=True, exist_ok=True)
    previous = {}
    if resume and (output / 'traces.json').exists():
        selection = read_json(output.parent / 'selection.json')
        if selection['fingerprint'] != fingerprint or selection['method'] != agent_method:
            raise ValueError('Cannot resume with changed benchmark, corpus or method')
        previous = {(row['method'], row['query_id']): row for row in read_json(output / 'traces.json')}
    traces, summaries = [], []
    for method in ['dense', 'bm25', 'hybrid']:
        answered = hits1 = hits3 = 0
        started = time.perf_counter()
        for case in selected:
            prior = previous.get((method, case['id']))
            needs_agent = agent and (split == 'dev' or method in {'dense', agent_method})
            if prior and prior.get('status') != 'error' and (not needs_agent or 'agent' in prior):
                traces.append(prior)
                if prior.get('answerable'):
                    answered += 1
                    hits1 += prior.get('hit1', False)
                    hits3 += prior.get('hit3', False)
                continue
            row = {'query_id': case['id'], 'method': method}
            if prior:
                row['previous_attempt'] = {'status': prior.get('status'), 'error_type': prior.get('error_type')}
            try:
                results = retriever.search(case['question'], method, top_k=6, filters=case.get('filter'))
                answerable = case.get('answerable', True)
                row.update(results=results, answerable=answerable)
                if answerable:
                    answered += 1
                    hit1 = evidence_complete(case, results[:1])
                    hit3 = evidence_complete(case, results[:3])
                    hits1 += hit1
                    hits3 += hit3
                    row.update(hit1=hit1, hit3=hit3)
                if needs_agent:
                    trace = run_agent(case['question'], retriever, method, filters=case.get('filter'))
                    row['agent'] = trace
                    if answerable:
                        row['context_evidence'] = evidence_complete(case, trace['context'])
            except Exception as error:
                row.update(status='error', error_type=type(error).__name__)
            traces.append(row)
            write_json(output / 'traces.json', traces)
        method_rows = [row for row in traces if row['method'] == method]
        agent_rows = [row for row in method_rows if 'agent' in row]
        context_rows = [row for row in agent_rows if row.get('answerable')]
        summaries.append({'method': method, 'split': split, 'answerable_retrieval_successes': answered,
                          'hit1': {'n': hits1, 'N': answered}, 'hit3': {'n': hits3, 'N': answered},
                          'context_evidence': {'n': sum(row.get('context_evidence', False) for row in context_rows),
                                               'N': len(context_rows)},
                          'agent_runs': len(agent_rows),
                          'total_agent_seconds': sum(row['agent'].get('seconds', 0) for row in agent_rows),
                          'tool_use': sum(any(e['event'] == 'search' for e in row['agent']['events']) for row in agent_rows),
                          'answer_quality_review': 'pending_human_review',
                          'seconds_this_invocation': time.perf_counter() - started,
                          'errors': sum(row.get('status') == 'error' for row in traces if row['method'] == method)})
    write_json(output / 'traces.json', traces)
    write_json(output / 'summary.json', summaries)
    reviews = [{'query_id': row['query_id'], 'method': row['method'], 'context_grade': None,
                'answer_grade': None, 'citation_support': None, 'abstention_correct': None,
                'review_status': 'pending_human_review'} for row in traces if 'agent' in row]
    write_json(output / 'reviews_pending.json', reviews)
    if split == 'dev':
        write_json(output.parent / 'selection.json', {'fingerprint': fingerprint, 'method': agent_method,
                   'provider':PROVIDER,'model':MODEL,
                   'status': 'configuration_frozen_before_test',
                   'note': 'Method explicitly chosen by CLI. Human analysis must justify selection.'})
    return summaries
