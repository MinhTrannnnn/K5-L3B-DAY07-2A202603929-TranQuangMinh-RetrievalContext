"""Local-only endpoint and citation binding regressions."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.common import api_client
from src.local_agent import run_local_agent


class LocalAgentTests(unittest.TestCase):
    def test_remote_model_endpoint_is_rejected(self):
        with patch('src.common.OLLAMA_URL','https://api.openai.com'):
            with self.assertRaises(ValueError):
                api_client()

    def test_client_never_receives_hosted_api_key(self):
        with patch('src.common.OpenAI') as constructor, patch('src.common.DefaultHttpxClient'):
            api_client()
        self.assertEqual(constructor.call_args.kwargs['api_key'],'ollama-local')
        self.assertIn('127.0.0.1',constructor.call_args.kwargs['base_url'])

    def run_fixture(self,reference,answer=None):
        item={'id':'actual-chunk-123','text':'Use with_mean=False.','source_id':'doc',
              'source':'manual','page':'1','metadata':{}}
        retriever=SimpleNamespace(corpus='day6',search=lambda *args:[item])
        decisions=[({'action':'search','query':'StandardScaler centering','answer':'','citations':[]},{}),
                   ({'action':'answer','query':'','answer':f'Use with_mean=False [{reference}].' if answer is None else answer,
                     'citations':[reference]}, {})]
        with patch('src.local_agent.local_json',side_effect=decisions):
            return run_local_agent('Disable centering?',retriever)

    def test_short_reference_is_bound_to_retrieved_chunk(self):
        result=self.run_fixture('S1')
        self.assertEqual(result['status'],'answered')
        self.assertEqual(result['citations'],['actual-chunk-123'])
        self.assertIn('[actual-chunk-123]',result['answer'])

    def test_unknown_reference_cannot_become_a_valid_citation(self):
        self.assertEqual(self.run_fixture('S9')['status'],'abstain')

    def test_empty_answer_cannot_become_an_answer_by_appending_citations(self):
        self.assertEqual(self.run_fixture('S1',answer='')['status'],'abstain')
