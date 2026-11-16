"""Regressions around citation guards and budgeted source expansion."""
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent import decode_answer, pack_results, run_agent
from src.evaluation import evaluate_cases, supported_gold
from src.common import read_json


class PipelineTests(unittest.TestCase):
    def test_multiple_final_messages_are_decoded_individually(self):
        text = json.dumps({'status': 'abstain', 'answer': 'Insufficient evidence', 'citations': []})
        message = SimpleNamespace(type='message', phase='final_answer',
                                  content=[SimpleNamespace(type='output_text', text=text)])
        response = SimpleNamespace(output=[message, message], output_text=text + text)
        self.assertEqual(decode_answer(response)['status'], 'abstain')

    def test_answer_without_search_is_rejected(self):
        response = SimpleNamespace(status='completed', model='test', usage={}, output=[],
            output_text=json.dumps({'status': 'answered', 'answer': 'Invented [x]', 'citations': ['x']}))
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: response))
        with patch('src.agent.PROVIDER','fixture'), patch('src.agent.api_client', return_value=client), patch('src.agent.record_usage', return_value={}):
            result = run_agent('Question?', SimpleNamespace(corpus='nist'))
        self.assertEqual(result['status'], 'abstain')
        self.assertEqual(result['citations'], [])

    def test_oversized_first_result_does_not_hide_following_evidence(self):
        rows = [{'id': 'large', 'text': 'large ' * 5000}, {'id': 'small', 'text': 'Useful evidence'}]
        selected = pack_results(rows, 100)
        self.assertEqual([row['id'] for row in selected], ['small'])

    def test_gold_requires_source_page_and_full_evidence(self):
        case = {'gold': [{'source_id': 'nist', 'page': 63, 'evidence': 'Approved Standard'}]}
        row = {'source_id': 'nist', 'page': '63', 'text': 'Status: Approved\nStandard'}
        self.assertTrue(supported_gold(case, row))
        self.assertFalse(supported_gold(case, {**row, 'page': '64'}))
        self.assertFalse(supported_gold(case, {**row, 'text': 'Approved'}))

    def test_resume_retries_only_failed_agent_case(self):
        item = {'id': 'e1', 'source_id': 'nist', 'page': '63', 'text': 'Approved Standard'}
        retriever = SimpleNamespace(chunks=[item], search=lambda *args, **kwargs: [item])
        case = {'id': 'q1', 'split': 'dev', 'question': 'Status?',
                'gold': [{'source_id': 'nist', 'page': 63, 'evidence': 'Approved Standard'}]}
        trace = {'context': [item], 'events': [{'event': 'search'}]}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'dev'
            with patch('src.evaluation.run_agent', side_effect=[trace, ValueError('retry'), trace]):
                evaluate_cases([case], retriever, 'dev', output, agent=True, agent_method='bm25')
            with patch('src.evaluation.run_agent', return_value=trace) as retry:
                summary = evaluate_cases([case], retriever, 'dev', output,
                                         agent=True, agent_method='bm25', resume=True)
            self.assertEqual(retry.call_count, 1)
            self.assertTrue(all(row['errors'] == 0 for row in summary))
            saved = read_json(output / 'traces.json')
            self.assertEqual(len(saved), 3)
            retried = [row for row in saved if row['method'] == 'bm25'][0]
            self.assertEqual(retried['previous_attempt']['error_type'], 'ValueError')
