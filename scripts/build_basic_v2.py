"""Fresh basic holdout after the documented abstention prompt change."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.common import DAY6,ROOT,digest,read_json,write_json


def main():
    target=ROOT/'benchmark/basic_v2_cases.json'
    if target.exists():
        raise FileExistsError('Fresh holdout already frozen')
    cases=[q for q in read_json(ROOT/'benchmark/basic_cases.json') if q['split']=='dev']
    units={row['page']:row for row in
           [json.loads(line) for line in (DAY6/'data/units.jsonl').read_text(encoding='utf-8').splitlines()]}
    definitions=[
        ('N01','Which PolynomialFeatures setting produces only feature interaction terms?',
         'Set interaction_only=True.','preprocessing:012',
         'In some cases, only interaction terms among features are required, and it can be gotten with the setting interaction_only=True :'),
        ('N02','In the ColumnTransformer example, what happens to the remaining rating columns by default?',
         'They are ignored/dropped with remainder="drop".','compose:005',
         "By default, the remaining rating columns are ignored ( remainder='drop' ):"),
        ('N03','What does RepeatedKFold do across repetitions?',
         'It repeats KFold n times with different splits in every repetition.','cross_validation:004',
         'RepeatedKFold repeats KFold \\(n\\) times, producing different splits in each repetition.'),
        ('N04','In cross_validate, what is the default return_train_score setting, why, and how can training scores be requested?',
         'It defaults to False to save computation time; set it to True for training scores.','cross_validation:003',
         'return_train_score is set to False by default to save computation time. To evaluate the scores on the training set as well you need to set it to True .'),
        ('N05','Which imputer parameter retains entirely empty features, and what constant is used in most cases?',
         'keep_empty_features=True; the constant is usually zero.','impute:003',
         'The parameter keep_empty_features offers the option to keep the empty features by imputing with a constant value. In most of the cases, this constant value is zero:')]
    for identity,question,answer,page,quote in definitions:
        unit=units[page]
        if ' '.join(quote.split()) not in ' '.join(unit['text'].split()):
            raise ValueError(f'Evidence does not resolve for {identity}')
        cases.append({'id':identity,'split':'test','question':question,'expected_answer':answer,
                      'answerable':True,'gold_source_id':unit['document_id'],'gold_page':page,
                      'evidence_quote':quote,'human_confirmation':'pending'})
    for identity,question,reason in [
        ('N06','What exact mean absolute error will SimpleImputer achieve on my private dataset, which is not included in these documents?',
         'Dataset-specific error cannot be determined without that dataset and experiment.'),
        ('N07','What exact per-hour cloud-computing price does RepeatedKFold charge according to these documents?',
         'The corpus supplies no RepeatedKFold cloud price or tariff.')]:
        cases.append({'id':identity,'split':'test','question':question,'answerable':False,
                      'expected_behavior':'search_and_abstain','expected_answer':'Not enough evidence to supply the requested exact value.',
                      'absence_reason':reason,'human_confirmation':'pending'})
    write_json(target,cases)
    write_json(ROOT/'benchmark/basic_v2_freeze.json',{
        'created_at':datetime.now(timezone.utc).isoformat(),'cases_sha256':digest(target.read_bytes()),
        'corpus_sha256':digest((DAY6/'data/units.jsonl').read_bytes()),
        'reason':'New test facts after generic abstention prompt fix. Prior test retained as historical evidence.'})
    print('24 basic v2 cases frozen')


if __name__=='__main__':main()
