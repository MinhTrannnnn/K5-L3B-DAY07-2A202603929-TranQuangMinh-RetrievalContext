"""Check benchmark structure, document coverage and absence-search evidence."""
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pymupdf
from src.common import ROOT,digest,read_json,write_json


def main():
    cases=read_json(ROOT/'benchmark/advanced_cases.json')
    manifest=read_json(ROOT/'data/processed/manifest_wdr.json')
    chunks=read_json(ROOT/'data/processed/wdr_chunks.json')
    with pymupdf.open(ROOT/'data/documents/WDR2024.pdf') as document:
        texts=[page.get_text(sort=True) for page in document]
        all_text='\n'.join(texts)
        audits=[]
        for case in cases:
            if case['answerable']:
                locations=[reference for group in case['evidence_groups'] for reference in group]
                assert all(1<=int(ref['page'])<=len(document) for ref in locations)
                indexed=all(any(str(chunk['page'])==ref['page'] for chunk in chunks) for ref in locations)
                assert indexed
                audits.append({'id':case['id'],'source_pages_valid':True,'source_pages_indexed':indexed,
                               'evidence_regions':[{'page':ref['page'],'label':ref['region_label']} for ref in locations],
                               'human_confirmation':False})
            else:
                terms={term:len(re.findall(re.escape(term),all_text,re.I)) for term in case['absence_search_terms']}
                audits.append({'id':case['id'],'absence_search_counts':terms,'reason':case['absence_reason'],
                               'note':'Term counts support the audit; they do not alone prove absence of semantically equivalent facts.',
                               'human_confirmation':False})
    counts=Counter(case['group'] for case in cases if case['split']=='test')
    minimum={'table':6,'image':5,'chart':5,'multi_hop':4,'text':4,'missing':6}
    assert all(counts[group]>=n for group,n in minimum.items())
    assert sum(case['split']=='dev' and case['answerable'] for case in cases)>=15
    assert sum(case['split']=='dev' and not case['answerable'] for case in cases)>=3
    assert len({case['id'] for case in cases})==len(cases)
    assert len({case['question'].casefold() for case in cases})==len(cases)
    assert manifest['pages']>=100 and manifest['visual_items'].get('chart',0)>0
    summary={'status':'structural_checks_passed','pdf_pages':manifest['pages'],
             'indexed_chunks':len(chunks),'test_groups':dict(counts),'dev_cases':18,'test_cases':30,
             'cases_sha256':digest((ROOT/'benchmark/advanced_cases.json').read_bytes()),
             'visual_counts':manifest['visual_items'],'audits':audits,
             'human_original_source_confirmation':'pending'}
    write_json(ROOT/'results/advanced/benchmark_audit.json',summary)
    print('Advanced benchmark structure, sources and index coverage verified.')


if __name__=='__main__':main()
