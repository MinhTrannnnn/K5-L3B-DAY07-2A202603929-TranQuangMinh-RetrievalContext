"""Run with python -m src from the Day 7 repository."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import (DAY6, MODEL, ROOT, api_client, read_json, record_usage, write_json)
from .storage import Retriever, connect, ingest_nist, restore_day6


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Day 7 Agentic RAG')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    extraction = commands.add_parser('extract')
    extraction.add_argument('--workers', type=int, default=3)
    extraction.add_argument('--corpus', choices=['nist', 'wdr'], default='nist')
    ingest = commands.add_parser('ingest')
    ingest.add_argument('--corpus', choices=['nist', 'wdr', 'day6'], required=True)
    query = commands.add_parser('query')
    query.add_argument('question')
    query.add_argument('--corpus', choices=['nist', 'wdr', 'day6'], default='day6')
    query.add_argument('--method', choices=['dense', 'bm25', 'hybrid'], default='bm25')
    query.add_argument('--top-k', type=int, default=6)
    query.add_argument('--budget', type=int, default=3000)
    query.add_argument('--max-rounds', type=int, default=1)
    query.add_argument('--no-vision', action='store_true')
    query.add_argument('--no-verify', action='store_true')
    query.add_argument('--output', type=Path)
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--cases', type=Path, required=True)
    evaluate.add_argument('--split', choices=['dev', 'test'], default='dev')
    evaluate.add_argument('--corpus', choices=['nist', 'wdr', 'day6'], default='day6')
    evaluate.add_argument('--agent', action='store_true')
    evaluate.add_argument('--resume', action='store_true')
    evaluate.add_argument('--output', type=Path, required=True)
    evaluate.add_argument('--method', choices=['dense', 'bm25', 'hybrid'], default='hybrid')
    arguments = parser.parse_args()
    if arguments.command == 'doctor':
        with connect() as conn:
            database = conn.execute("SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname='vector'").fetchone()
        from .local_agent import local_json
        response,usage = local_json([{'role':'user','content':'Return {"ok":true}.'}],
                                   {'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok']},64)
        result = {'database': database, 'model': MODEL, 'provider':'ollama', 'response': response,
                  'python': sys.version, 'day6_path_exists': DAY6.exists()}
        write_json(ROOT / 'results/local/doctor.json', result)
    elif arguments.command == 'extract':
        from .extraction import extract_pdf
        result = extract_pdf(arguments.workers, arguments.corpus)
    elif arguments.command == 'ingest':
        result = restore_day6() if arguments.corpus == 'day6' else ingest_nist(arguments.corpus)
        write_json(ROOT / f'results/local/ingest_{arguments.corpus}.json', result)
    elif arguments.command == 'query':
        from .agent import run_agent
        output = arguments.output or ROOT / 'results/local/queries' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json')
        try:
            result = run_agent(arguments.question, Retriever(arguments.corpus), arguments.method,
                               arguments.top_k, arguments.budget, arguments.max_rounds,
                               vision=not arguments.no_vision, verify=False)
        except Exception as error:
            write_json(output, {'status': 'error', 'question': arguments.question,
                               'error_type': type(error).__name__,
                               'http_status': getattr(error, 'status_code', None)})
            raise
        write_json(output, result)
        print(json.dumps({'status': result['status'], 'answer': result['answer'], 'trace': str(output)}, ensure_ascii=False))
        return
    else:
        from .evaluation import evaluate_cases
        result = evaluate_cases(read_json(arguments.cases), Retriever(arguments.corpus), arguments.split,
                                arguments.output, arguments.agent, arguments.method, arguments.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Avoid printing connection URLs, API keys or response bodies in shared logs.
        print(json.dumps({'status': 'error', 'error_type': type(error).__name__,
                          'http_status': getattr(error, 'status_code', None)}), file=sys.stderr)
        raise SystemExit(1)
