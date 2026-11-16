"""Run the local Day 6 based lab, selecting retrieval on dev before test."""
import os
import subprocess
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read_json,write_json


def run(*arguments):
    subprocess.run([sys.executable,'-X','utf8',*arguments],cwd=ROOT,check=True)


def main():
    run('-m','src','ingest','--corpus','day6')
    base=ROOT/'results/local/final'
    common=['-m','src','evaluate','--cases','benchmark/local_basic_cases.json','--corpus','day6','--agent']
    dev=[*common,'--split','dev','--method','bm25','--output',str(base/'dev')]
    if (base/'selection.json').exists():dev.append('--resume')
    run(*dev)
    summaries=read_json(base/'dev/summary.json')
    if any(row['errors'] for row in summaries):
        raise RuntimeError('Resolve recorded dev errors before selecting a test configuration')
    best=max(summaries,key=lambda row:(row['hit3']['n'],row['hit1']['n'],-row['total_agent_seconds']))
    selection=read_json(base/'selection.json')
    selection.update(method=best['method'],rule='Dev Evidence Hit@3, then Hit@1, then lower agent time.')
    write_json(base/'selection.json',selection)
    test=[*common,'--split','test','--method',best['method'],'--output',str(base/'test')]
    if (base/'test/traces.json').exists():test.append('--resume')
    run(*test)
    print('Local basic experiment complete. Results: results/local/final')


if __name__=='__main__':main()
