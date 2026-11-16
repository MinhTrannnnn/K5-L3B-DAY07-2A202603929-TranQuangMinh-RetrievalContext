"""Resume the remaining paid work once API credit is available."""
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import ROOT,read_json,write_json


def run(*arguments):
    print('Running', ' '.join(arguments),flush=True)
    subprocess.run([sys.executable,*arguments],cwd=ROOT,check=True)


def main():
    run('-m','src','doctor')
    runner='scripts/run_advanced.py'
    run(runner,'--suite','basic_v2','--split','dev','--resume')
    run(runner,'--suite','basic_v2','--split','test','--resume')
    run(runner,'--split','dev','--benchmark-version','2','--reuse-from','results/advanced','--resume')
    run(runner,'--split','dev','--benchmark-version','2','--experiment','verified',
        '--verify','--top-k','10','--budget','10000','--max-rounds','8',
        '--reuse-from','results/advanced_verified','--resume')
    candidates=[]
    for experiment,folder in [('main','advanced_v2'),('verified','advanced_v2_verified')]:
        selection=read_json(ROOT/f'results/{folder}/selection.json')
        if selection.get('status')!='selected_on_complete_dev':
            raise ValueError('Dev reviews incomplete; restore API credit then resume.')
        for summary in read_json(ROOT/f'results/{folder}/dev/summary.json'):
            if summary['method']==selection['method']:
                candidates.append((summary,experiment,folder,selection))
    best=max(candidates,key=lambda item:(item[0]['passed'],item[0]['hit3']['n'],-item[0]['total_agent_seconds']))
    summary,experiment,folder,selection=best
    write_json(ROOT/'results/final_selection.json',{'experiment':experiment,'folder':folder,
               'method':selection['method'],'pipeline':selection['pipeline'],
               'dev_review_passed':summary['passed'],'dev_review_total':summary['total'],
               'selected_before_test_at':datetime.now(timezone.utc).isoformat(),
               'selection_rule':'Complete dev assistant-review pass count, then Hit@3, then total agent time.',
               'human_confirmation':False})
    args=[runner,'--split','test','--benchmark-version','2','--resume']
    if experiment=='verified':
        args.extend(['--experiment','verified','--verify','--top-k','10','--budget','10000','--max-rounds','8'])
    run(*args)
    run('scripts/summarize_usage.py')
    run('scripts/build_review_workspace.py')
    write_json(ROOT/'results/execution_status.json',{'state':'automated_runs_complete',
               'advanced_test_executed':True,'selected_folder':folder,
               'human_confirmation':False,'remaining':['Student original-source review and confirmation.']})
    print('Automated runs finished. Open results/review/index.html for original-source confirmation.')


if __name__=='__main__':main()
