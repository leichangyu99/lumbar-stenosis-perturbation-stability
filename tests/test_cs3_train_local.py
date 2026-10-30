"""Local smoke of cs3_train.main on 8 real development patients (random labels), 1 epoch, 1 seed, A0 + A2."""
import contextlib, json, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import cs_common as C, cs3_train as T
PRED = HERE.parent.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
rows = json.loads(PRED.read_text(encoding='utf-8'))['predictions']
diag = [r['study_id'] for r in rows if r['role'] == 'diagnostic_train'][:8]; sel = [r['study_id'] for r in rows if r['role'] == 'model_selection'][:4]
tmp = Path(tempfile.mkdtemp()); (tmp / 'cs/e0').mkdir(parents=True); job = tmp / 'job'; job.mkdir()
json.dump({'patient_ids': {'diagnostic_train': diag, 'model_selection': sel}}, open(tmp / 'cs/e0/strategy_split.json', 'w'))
rng = np.random.default_rng(0); json.dump({'labels': {s: [int(x) for x in rng.integers(-1, 3, 5)] for s in diag}}, open(tmp / 'labels.json', 'w'))
json.dump({'files_sha256': {}}, open(job / 'INVENTORY.json', 'w'))
T.HERE = job; T.CS = tmp / 'cs'; T.DATA = Path('data/LumbarDISC/dataset'); T.PRED = PRED; T.WEIGHTS = HERE.parent / 'weights_ma'
T.DIAG_LABELS = tmp / 'labels.json'; T.DIAG_LABELS_SHA = C.sha(tmp / 'labels.json'); T.EPOCHS = 1; T.SEEDS = (1,); T.GpuLock = lambda n: contextlib.nullcontext()
T.labels_from_train_csv = lambda ids: {s: [int(x) for x in rng.integers(-1, 3, 5)] for s in ids}
sys.argv = ['x', '--archs', 'A0', 'A2', '--tag', 'T']; T.main()
r = json.load(open(job / 'out_T/TRAIN_RESULT.json')); print(r['status'], {k: (round(v['dev_RPS_model_selection'], 4), round(v['seconds_at_end'], 1)) for k, v in r['fits'].items()}, r.get('peak_gpu_gib'))
assert r['status'] == 'completed'; print('cs3 train smoke passed')
