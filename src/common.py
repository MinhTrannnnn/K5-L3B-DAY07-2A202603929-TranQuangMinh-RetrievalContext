"""Configuration, reproducible caches and API usage records."""
import hashlib
import json
import os
import threading
from urllib.parse import urlparse
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import DefaultHttpxClient, OpenAI

ROOT = Path(__file__).resolve().parents[1]
DAY6 = ROOT.parents[1] / 'Day6' / 'K5-L3B-DAY06-TranQuangMinh-2A202603929-DataIngestion'
load_dotenv(ROOT / '.env')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
PROVIDER = 'ollama'
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434').rstrip('/')
MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5:3b')
VISION_MODEL = os.getenv('OLLAMA_VISION_MODEL', 'qwen3-vl:2b')
EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
DIMENSIONS = 1024
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab')
PDF = ROOT / 'NIST.SP.500-291r2.pdf'
CORPORA = json.loads((ROOT / 'data/corpora.json').read_text(encoding='utf-8'))
DAY6_CONFIG = json.loads((DAY6 / 'config/experiment.json').read_text(encoding='utf-8'))
DAY6_EXPERIMENT = json.loads((DAY6 / 'results/selection.json').read_text(encoding='utf-8'))['experiment']
_usage_lock = threading.Lock()


def digest(value):
    payload = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def decode_json_response(response):
    """Decode the final message, since SDK output_text concatenates messages."""
    messages = [item for item in response.output if item.type == 'message'
                and getattr(item, 'phase', None) != 'commentary']
    if messages:
        content = ''.join(part.text for part in messages[-1].content if part.type == 'output_text')
        return json.loads(content)
    return json.loads(response.output_text)


def api_client():
    parsed = urlparse(OLLAMA_URL)
    if parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('OLLAMA_URL must point to a local loopback HTTP server')
    if any('cloud' in name.lower() for name in [MODEL, VISION_MODEL, EMBEDDING_MODEL]):
        raise ValueError('Only local model tags are allowed')
    return OpenAI(base_url=OLLAMA_URL + '/v1/', api_key='ollama-local', timeout=300,
                  max_retries=0, http_client=DefaultHttpxClient(trust_env=False))


def record_usage(operation, model, usage, **details):
    row = {'at': datetime.now(timezone.utc).isoformat(), 'operation': operation,
           'model': model, 'provider': PROVIDER, 'api_cost_usd': 0,
           'usage': usage.model_dump() if hasattr(usage, 'model_dump') else usage,
           **details}
    path = ROOT / 'results/local/usage.jsonl'
    with _usage_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row
