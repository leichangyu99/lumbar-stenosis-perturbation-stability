"""CS20261015-CPU: local predictions of the six multi-architecture graders (A0-A5, 3 seeds) on the RSNA validation set
(model_selection 197) for the 27 frozen evaluation views, with the same code path as cs3_eval.predict_set.  Hash-locked."""
import hashlib, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cs_engine as E, cs2_engine as E2, cs3_arch as A  # noqa: E402

DATA = Path('data/LumbarDISC/dataset')
PRED = HERE.parent.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
MODELS = HERE / 'local_models'; W = HERE.parent / 'weights_ma'; OUT = HERE / 'mirror/val'
SEEDS = (20261012, 20261013, 20261014)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True); out = OUT / 'PRED_validation.npz'; assert not out.exists()
    split = json.loads((HERE.parent / 'configs/strategy_split.json').read_text(encoding='utf-8-sig'))['patient_ids']; ids = split['model_selection']
    rows = {r['study_id']: r for r in json.loads(PRED.read_text(encoding='utf-8'))['predictions'] if r['role'] == 'model_selection'}; assert set(rows) == set(ids)
    fits = {}
    for tag in 'ABC': fits.update({k: (tag, v) for k, v in json.loads((MODELS / ('out_' + tag) / 'TRAIN_RESULT.json').read_text())['fits'].items()})
    torch.set_num_threads(4); models = []; priors = {}
    for arch in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
        for s in SEEDS:
            n = '%s_%d' % (arch, s); tag, f = fits[n]; p = MODELS / ('out_' + tag) / n / 'FINAL_EPOCH12.pt'; assert sha(p) == f['checkpoint_sha256'], n
            ck = torch.load(p, map_location='cpu', weights_only=True)
            net = A.Grader(arch, W / A.weight_file(arch)).cuda(); net.load_state_dict(ck['state_dict']); net.eval(); models.append((n, net)); priors[n] = ck['priors']
    views = E.evaluation_views(); vn = [v['name'] for v in views]
    probs = np.zeros((len(models), len(ids), len(vn), 5, 3)); vis = np.zeros((len(ids), len(vn), 5), dtype=bool)
    for i, sid in enumerate(ids):
        pv = A.patient_views(E2.Patient(DATA, rows[sid], sid), views)
        X = torch.stack([pv[n][0] for n in vn]).cuda(); M = torch.stack([pv[n][1] for n in vn]).cuda(); V = M.any(-1); vis[i] = V.cpu().numpy()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for m_i, (n, net) in enumerate(models):
                z = torch.cat([net(X[s:s + 9], M[s:s + 9]).float() for s in range(0, len(vn), 9)]).softmax(-1).double()
                z = torch.where(V[..., None], z, priors[n].double().cuda()[None].expand(len(vn), -1, -1)); probs[m_i, i] = z.cpu().numpy()
        assert np.isfinite(probs[:, i]).all()
        if (i + 1) % 25 == 0: print(json.dumps({'patients': i + 1, 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    np.savez(out, probs=probs, visible=vis, models=np.array([n for n, _ in models]), patients=np.array(ids), views=np.array(vn))
    (OUT / 'PREDICTIONS_LOCK.json').write_text(json.dumps({'validation': sha(out), 'source_sha256': sha(__file__), 'seconds': time.time() - t0,
                                                           'device': torch.cuda.get_device_name(0)}, indent=1), encoding='utf-8')
    print('locked', sha(out))


if __name__ == '__main__':
    main()
