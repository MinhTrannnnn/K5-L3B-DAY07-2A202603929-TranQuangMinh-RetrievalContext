"""Create an offline source-review workspace without claiming human confirmation."""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,digest,read_json


def main():
    final=ROOT/'results/final_selection.json'
    if final.exists():
        selection=read_json(final)
        folder=ROOT/'results'/selection['folder']
        method=selection['method']
        split='test'
        cases=read_json(ROOT/'benchmark/advanced_cases_v2.json')
        title='Đối chiếu 30 câu test nâng cao'
    else:
        folder=ROOT/'results/advanced'
        method=read_json(folder/'selection.json')['method']
        split='dev'
        cases=read_json(ROOT/'benchmark/advanced_cases.json')
        selection={'pipeline':read_json(folder/'selection.json')['pipeline']}
        title='Đối chiếu dev nâng cao — test chưa chạy'
    records=[]
    for case in cases:
        if case['split']!=split:continue
        path=folder/split/method/f'{case["id"]}.json'
        if not path.exists():continue
        review_path=path.with_name(f'{case["id"]}.review.json')
        records.append({'case':case,'trace':read_json(path),
                        'review':read_json(review_path) if review_path.exists() else {'review_error':'not_run'},
                        'trace_path':str(path.relative_to(ROOT))})
    parts=[]
    for record in records:
        case,trace,review=record['case'],record['trace'],record['review']
        identity=case['id'];passed=review.get('passed',False)
        pages=sorted({int(r['page']) for group in case.get('evidence_groups',[]) for r in group})
        links=' '.join(f'<a href="../../data/documents/WDR2024.pdf#page={p}" target="_blank">PDF trang {p}</a>' for p in pages)
        images=''.join(f'<img loading="lazy" src="../../data/processed/review_pages/wdr/page-{p:03}.png" alt="Trang PDF {p}">' for p in pages
                       if (ROOT/f'data/processed/review_pages/wdr/page-{p:03}.png').exists())
        parts.append(f'''<article data-id="{identity}"><h2>{identity} · {html.escape(case.get('group',''))}</h2>
<p>{html.escape(case['question'])}</p><p><strong>Đáp án tham chiếu</strong> {html.escape(case.get('expected_answer',''))}</p>
<p><strong>Agent</strong> {html.escape(trace.get('answer',trace.get('status','')))}</p>
<p><strong>Nhận xét hỗ trợ</strong> {html.escape(review.get('reason',review.get('review_error','Chưa chấm')))}</p>
<p>{links} · <a href="../../{record['trace_path']}" target="_blank">Trace đầy đủ</a></p>
<details><summary>Xem bằng chứng gốc</summary>{images}</details>
<label>Kết quả <select class="pass"><option value="true" {'selected' if passed else ''}>Đạt</option><option value="false" {'' if passed else 'selected'}>Không đạt</option></select></label>
<label>Lý do sau đối chiếu<textarea class="reason">{html.escape(review.get('reason',''))}</textarea></label>
<label><input type="checkbox" class="confirmed"> Tôi đã đọc câu trả lời và đối chiếu nguồn gốc cho câu này</label></article>''')
    metadata=json.dumps({'cases_hash':digest(cases),'pipeline_hash':digest(selection['pipeline']),
                         'split':split,'expected_ids':[r['case']['id'] for r in records]},ensure_ascii=False).replace('<','\\u003c')
    document='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Đối chiếu benchmark</title><style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;background:#f4f5f6;color:#17212b}article{background:white;border:1px solid #d7dce0;border-radius:8px;padding:24px;margin:24px 0}h2{font-size:19px}label{display:block;margin:14px 0}textarea{display:block;width:98%;min-height:80px;padding:8px}img{width:100%;height:auto;margin:12px 0}button{padding:12px 20px;cursor:pointer}input[type=text]{padding:10px;width:280px}a{color:#174f8a}header{background:#fff;padding:24px;border-radius:8px}#status{font-weight:600}</style>
<header><h1>'''+html.escape(title)+'''</h1><p>Nhãn hiện tại do trợ lý hỗ trợ chấm. Điền tên và xác nhận từng câu sau khi đọc nguồn gốc. Bạn có thể sửa kết quả và lý do.</p><label>Người đối chiếu <input id="reviewer" type="text" placeholder="Họ và tên"></label><p id="status"></p><button id="export">Xuất nhãn đã xác nhận</button></header>'''+''.join(parts)+'''
<script type="application/json" id="metadata">'''+metadata+'''</script><script>
const records=[...document.querySelectorAll('article')];
function update(){document.getElementById('status').textContent=records.filter(r=>r.querySelector('.confirmed').checked).length+' / '+records.length+' câu đã xác nhận';}
document.addEventListener('change',update);update();
document.getElementById('export').onclick=()=>{const name=document.getElementById('reviewer').value.trim();
if(!name||records.some(r=>!r.querySelector('.confirmed').checked||!r.querySelector('.reason').value.trim())){alert('Điền tên, lý do và xác nhận từng câu sau khi đối chiếu.');return;}
const metadata=JSON.parse(document.getElementById('metadata').textContent);
const data={...metadata,reviewer:name,reviewed_at:new Date().toISOString(),items:records.map(r=>({query_id:r.dataset.id,passed:r.querySelector('.pass').value==='true',reason:r.querySelector('.reason').value.trim(),human_confirmed:true}))};
const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='human_reviews.json';a.click();URL.revokeObjectURL(url);};</script></html>'''
    path=ROOT/'results/review/index.html';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(document,encoding='utf-8')
    print(path)


if __name__=='__main__':main()
