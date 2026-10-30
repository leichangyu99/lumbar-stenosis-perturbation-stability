"""CS20261013-EXT step 2: all frozen models predict the 27 evaluation views + 8 TTA geometric jitters of the
identity view for de-identified dataset 3 (local RTX5060).  S0 = CS20261012 E2 STD (frozen-prefix ResNet18, layer3
maps from the frozen engine); A0-A5 = multi-architecture graders (raw crops). Reference grades were completed before
prediction generation. Local only; nothing is uploaded."""
import hashlib, io, json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cs_common as Cm  # noqa: E402
import cs_engine as E, cs2_engine as E2, cs3_arch as A  # noqa: E402

ROOT = Path('data/hospital/images_batch1')
WORK = Path('data/hospital/work/external_dataset3')
MODELS = HERE / 'local_models'
W_MA = HERE.parent / 'weights_ma'
Cm.WEIGHTS = HERE.parent / 'weights/resnet18-f37072fd.pth'; Cm.HELPERS = HERE.parent
SEEDS = (20261012, 20261013, 20261014)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    t0 = time.time(); out = WORK / 'PRED_dataset3.npz'; assert not out.exists()
    loc = json.loads((WORK / 'LOCALIZATION.json').read_text(encoding='utf-8')); rows = {r['study_id']: r for r in loc['predictions']}; codes = sorted(rows)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    views = E.evaluation_views(); vn = [v['name'] for v in views]; keys = [(v, None) for v in vn] + [('I', m) for m in range(8)]
    # ---- models ----
    R = Cm.Recipe(torch, np); e2 = json.loads((MODELS / 'e2/E2_RESULT.json').read_text()); names = []; std_nets = []; ma_nets = []
    for s in SEEDS:
        n = 'S0_%d' % s; p = MODELS / 'e2' / ('STD_%d' % s) / 'FINAL_EPOCH12.pt'; assert sha(p) == e2['fits']['STD_%d' % s]['checkpoint_sha256'], n
        ck = torch.load(p, map_location='cpu', weights_only=True)
        net = R.Diagnosis(R.backbone.layer4, R.SagHead().cuda(), torch.zeros(512).cuda(), torch.ones(512).cuda(), True).cuda(); net.load_state_dict(ck['state_dict']); net.eval()
        std_nets.append((n, net, ck['priors'])); names.append(n)
    tr = {}
    for tag in 'ABC': tr.update(json.loads((MODELS / ('out_' + tag) / 'TRAIN_RESULT.json').read_text())['fits'] | {'_tag_' + k: tag for k in json.loads((MODELS / ('out_' + tag) / 'TRAIN_RESULT.json').read_text())['fits']})
    for arch in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
        for s in SEEDS:
            n = '%s_%d' % (arch, s); p = MODELS / ('out_' + tr['_tag_' + n]) / n / 'FINAL_EPOCH12.pt'; assert sha(p) == tr[n]['checkpoint_sha256'], n
            ck = torch.load(p, map_location='cpu', weights_only=True)
            net = A.Grader(arch, W_MA / A.weight_file(arch)).cuda(); net.load_state_dict(ck['state_dict']); net.eval(); ma_nets.append((n, net, ck['priors'])); names.append(n)
    _, prefix = E.load_prefix(Cm.WEIGHTS, 'cuda')
    probs = np.zeros((len(names), len(codes), len(keys), 5, 3)); vis = np.zeros((len(codes), len(keys), 5), dtype=bool)
    for i, c in enumerate(codes):
        P = E2.Patient(ROOT, rows[c], c)
        jit = [E2.compose(E.IDENTITY | {'name': 'I'}, E2.tta_jitter(c, m)) for m in range(8)]
        # layer3 maps (S0) via the frozen engine; raw crops (A*) via cs3_arch, both from the same decoded patient
        lmaps = {}
        for v in views:
            if v['family'] == 'D': continue
            lmaps[v['name']] = P.view(v, prefix)
        for v in views:
            if v['family'] == 'D': lmaps[v['name']] = E2.drop(*lmaps['I'], v['drop'])
        for m, j in enumerate(jit): lmaps[('I', m)] = P.view(j, prefix)
        raw = A.patient_views(P, views)
        for m, j in enumerate(jit): raw[('I', m)] = A.patient_raw(P, j)
        kk = [k[0] if k[1] is None else k for k in keys]
        for q, k in enumerate(kk): vis[i, q] = (lmaps[k][1].any(-1)).numpy()
        with torch.no_grad():
            for m_i, (n, net, pri) in enumerate(std_nets):
                for s0 in range(0, len(kk), 7):
                    b = kk[s0:s0 + 7]; z = net(torch.stack([lmaps[k][0] for k in b]).cuda(), torch.stack([lmaps[k][1] for k in b]).cuda()).softmax(-1).double().cpu()
                    vv = torch.stack([lmaps[k][1] for k in b]).any(-1); z = torch.where(vv[..., None], z, pri.double()[None].expand(len(z), -1, -1)); probs[m_i, i, s0:s0 + len(b)] = z.numpy()
            X = torch.stack([raw[k][0] for k in kk]).cuda(); M = torch.stack([raw[k][1] for k in kk]).cuda(); V = M.any(-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for m_i, (n, net, pri) in enumerate(ma_nets, start=len(std_nets)):
                    z = torch.cat([net(X[s:s + 7], M[s:s + 7]).float() for s in range(0, len(kk), 7)]).softmax(-1).double()
                    z = torch.where(V[..., None], z, pri.double().cuda()[None].expand(len(kk), -1, -1)); probs[m_i, i] = z.cpu().numpy()
        assert np.isfinite(probs[:, i]).all()
        if (i + 1) % 10 == 0: print(json.dumps({'patients': i + 1, 'elapsed_s': round(time.time() - t0, 1)}), flush=True)
    np.savez(out, probs=probs, visible=vis, models=np.array(names), patients=np.array(codes), keys=np.array(['%s|%s' % k for k in keys]))
    lock = {'predictions_sha256': sha(out), 'localization_sha256': sha(WORK / 'LOCALIZATION.json'), 'labels_read_before_lock': False, 'models': names,
            'source_sha256': sha(__file__), 'seconds': time.time() - t0, 'device': torch.cuda.get_device_name(0)}
    (WORK / 'PREDICTIONS_LOCK.json').write_text(json.dumps(lock, indent=1) + '\n', encoding='utf-8'); print(json.dumps({k: v for k, v in lock.items() if k != 'models'}))


if __name__ == '__main__':
    main()
