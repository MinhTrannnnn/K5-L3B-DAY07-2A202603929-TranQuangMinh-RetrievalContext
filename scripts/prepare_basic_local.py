"""Download the pinned public BGE weights once; inference remains local."""
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import DAY6_CONFIG, ROOT
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':
    SentenceTransformer(DAY6_CONFIG['model'], revision=DAY6_CONFIG['model_revision'],
                        device='cpu', cache_folder=str(ROOT / '.cache/huggingface'))
    print('Pinned BGE model cached. Next: python -m src ingest --corpus day6')
