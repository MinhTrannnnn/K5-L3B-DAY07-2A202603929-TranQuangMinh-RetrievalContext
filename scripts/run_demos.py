"""Real API integration demonstrations; these are development examples, not a holdout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import run_agent
from src.common import ROOT, write_json
from src.storage import Retriever


def main():
    nist = Retriever('nist')
    cases = [
        ('nist_text', 'dense', 'Theo tài liệu, cloud computing có những service model nào?', nist),
        ('nist_bm25', 'bm25', 'Theo Table 5, SAML do SDO nào ban hành? Kiểm tra ảnh bảng.', nist),
        ('nist_diagram', 'hybrid', 'Mở Figure 1 và cho biết Cloud Carrier nằm ở đâu, kết nối những bên nào?', nist),
        ('nist_abstain', 'hybrid', 'Theo tài liệu này, giá gói AWS EC2 cho năm 2030 chính xác là bao nhiêu?', nist),
        ('day6_dense', 'dense', 'How can centering be disabled while retaining scaling in StandardScaler?', Retriever('day6')),
    ]
    summaries = []
    for name, method, question, retriever in cases:
        path = ROOT / 'results/demos' / f'{name}.json'
        try:
            trace = run_agent(question, retriever, method=method)
            write_json(path, trace)
            summary = {'name': name, 'status': trace['status'], 'answer': trace['answer'],
                       'method': method, 'seconds': trace['seconds'],
                       'events': [event['event'] for event in trace['events']],
                       'trace': str(path.relative_to(ROOT))}
        except Exception as error:
            summary = {'name': name, 'status': 'error', 'error_type': type(error).__name__}
            write_json(path, summary)
        summaries.append(summary)
        print(name, summary['status'], flush=True)
        write_json(ROOT / 'results/demos/summary.json', summaries)


if __name__ == '__main__':
    main()
