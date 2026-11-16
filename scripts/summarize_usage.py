"""Estimate API cost from recorded usage; prices are explicit and auditable."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import MODEL, PROVIDER, ROOT, write_json


def main():
    if PROVIDER=='ollama':
        path=ROOT/'results/local/usage.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        summary={'provider':'ollama','model':MODEL,'calls':len(rows),
                 'input_tokens':sum(row['usage'].get('input_tokens',0) for row in rows),
                 'output_tokens':sum(row['usage'].get('output_tokens',0) for row in rows),
                 'api_cost_usd':0,'note':'Local inference. CPU/GPU and electricity costs are not monetized.'}
        write_json(ROOT/'results/local/cost_summary.json',summary)
        print(json.dumps(summary,indent=2))
        return
    totals = defaultdict(lambda: {'calls': 0, 'input_tokens': 0, 'cached_input_tokens': 0,
                                  'output_tokens': 0, 'estimated_usd': 0.0})
    for line in (ROOT / 'results/api_usage.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        model, usage = row['model'], row['usage']
        entry = totals[model]
        entry['calls'] += 1
        incoming = usage.get('input_tokens', usage.get('prompt_tokens', 0))
        outgoing = usage.get('output_tokens', 0)
        cached = usage.get('input_tokens_details', {}).get('cached_tokens', 0)
        entry['input_tokens'] += incoming
        entry['output_tokens'] += outgoing
        entry['cached_input_tokens'] += cached
        if model.startswith('gpt-5.4-mini'):
            entry['estimated_usd'] += ((incoming - cached) * 0.75 + cached * 0.075 + outgoing * 4.50) / 1_000_000
        elif model == 'text-embedding-3-large':
            entry['estimated_usd'] += incoming * 0.13 / 1_000_000
        else:
            entry['price_unknown'] = True
    summary = {'models': dict(totals), 'estimated_total_usd': sum(r['estimated_usd'] for r in totals.values()),
               'price_checked_date': '2026-11-16',
               'sources': ['https://developers.openai.com/api/docs/models/gpt-5.4-mini',
                           'https://developers.openai.com/api/docs/models/text-embedding-3-large'],
               'note': 'Estimate from logged successful responses, excluding unlogged failures, local CPU and network.'}
    write_json(ROOT / 'results/cost_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
