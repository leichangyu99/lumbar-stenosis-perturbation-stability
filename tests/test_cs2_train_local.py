"""Local smoke test of cs2_train.main on synthetic cache/AugMix files (1 epoch, 1 seed)."""
import contextlib, hashlib, io, json, sys, tempfile
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(HERE))
import cs_common as C, cs2_train as T
C.WEIGHTS = HERE.parent / 'weights/resnet18-f37072fd.pth'; C.HELPERS = HERE.parent
tmp = Path(tempfile.mkdtemp()); g = torch.Generator().manual_seed(0); ids = ['P%02d' % i for i in range(40)]
(tmp / 'cache').mkdir(); (tmp / 'cs/e0').mkdir(parents=True); aug = tmp / 'cs/v2/augmix'; (aug / 'e01').mkdir(parents=True); job = tmp / 'job'; job.mkdir()
json.dump({'patient_ids': {'diagnostic_train': ids}}, open(tmp / 'cs/e0/strategy_split.json', 'w'))
labels = {s: [int(x) for x in torch.randint(-1, 3, (5,), generator=g)] for s in ids}; json.dump({'labels': labels}, open(tmp / 'labels.json', 'w'))
index = {}
for s in ids:
    m = torch.randn(5, 3, 256, 14, 14, generator=g).abs(); k = torch.rand(5, 3, generator=g) > .2
    torch.save({'maps': m, 'mask': k, 'study_id': s, 'role': 'diagnostic_train'}, tmp / 'cache' / (s + '.pt'))
    buf = io.BytesIO(); torch.save({'study_id': s, 'epoch': 1, 'k1': {'maps': m * 1.1, 'mask': k}, 'k2': {'maps': m * .9, 'mask': k}}, buf)
    (aug / 'e01' / (s + '.pt')).write_bytes(buf.getvalue()); index['e01/%s.pt' % s] = hashlib.sha256(buf.getvalue()).hexdigest()
json.dump(index, open(aug / 'FILES_SHA256.json', 'w'))
json.dump({'status': 'completed', 'files_index_sha256': C.sha(aug / 'FILES_SHA256.json')}, open(aug / 'VIEWS_RESULT.json', 'w'))
json.dump({'files_sha256': {}}, open(job / 'INVENTORY.json', 'w'))
T.HERE = job; T.OUT = job / 'out'; T.CS = tmp / 'cs'; T.AUG = aug; T.CACHE = tmp / 'cache'; T.DIAG_LABELS = tmp / 'labels.json'; T.DIAG_LABELS_SHA = C.sha(tmp / 'labels.json')
T.SEEDS = (20261012,); T.EPOCHS = 1; T.GpuLock = lambda name: contextlib.nullcontext()
T.main(); r = json.load(open(job / 'out/TRAIN_RESULT.json')); print(r['status'], r['fits']); assert r['status'] == 'completed' and len(r['fits']) == 2
print('cs2 train smoke passed')
