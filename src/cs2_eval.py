"""CS20261013 evaluation (PLAN.md sections 3.2 and 5).  --set selection = development (model_selection 197,
labels previously read in CS20261012 E5); --set policy = one-time independent evaluation (policy_train 474,
reference grades were completed before prediction generation).
Phase A: 11 evaluation views + 8 TTA jitters each are generated on the fly and predicted by 15 frozen
models (CS20261012 E2 STD/AUG/CONS x3, CS20261013 AUGMIX/AUGCE x3); only predictions are kept and locked.
Phase B: reference grades used for R1/R2/R3 statistics with patient bootstrap 10000, seed 20261013."""
import argparse, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, DATA, PRED, SEEDS, WEIGHTS, Clock, GpuLock, Recipe, labels_from_train_csv, load, require, rps_patient, save, severe_miss, sha  # noqa: E402

OUT = HERE / 'out'
SCHEMES = ('STD', 'AUG', 'CONS', 'AUGMIX', 'AUGCE')
B, BSEED = 10000, 20261013


def phase_a(which, clock, ev, np, torch):
    import cs_engine as E, cs2_engine as E2
    split = load(CS / 'e0/strategy_split.json')['patient_ids']; role = {'selection': 'model_selection', 'policy': 'policy_train'}[which]
    ids = split[role]; rows = {r['study_id']: r for r in load(PRED)['predictions'] if r['role'] == role}; require(set(rows) == set(ids), 'Localisation rows missing')
    e2 = load(CS / 'e2/out/E2_RESULT.json'); v2 = load(CS / 'v2train/out/TRAIN_RESULT.json'); require(e2['status'] == v2['status'] == 'completed', 'Training incomplete')
    R = Recipe(torch, np); names = []; models = []
    for scheme in SCHEMES:
        for seed in SEEDS:
            n = '%s_%d' % (scheme, seed); base, res = (CS / 'e2/out', e2) if scheme in ('STD', 'AUG', 'CONS') else (CS / 'v2train/out', v2)
            p = base / n / 'FINAL_EPOCH12.pt'; require(sha(p) == res['fits'][n]['checkpoint_sha256'], 'Checkpoint changed ' + n)
            ck = torch.load(p, map_location='cpu', weights_only=True)
            net = R.Diagnosis(R.backbone.layer4, R.SagHead().cuda(), torch.zeros(512).cuda(), torch.ones(512).cuda(), True).cuda(); net.load_state_dict(ck['state_dict']); net.eval()
            names.append(n); models.append((net, ck['priors']))
    _, prefix = E.load_prefix(WEIGHTS, 'cuda')
    vnames = [v['name'] for v in E2.eval_views()]; keys = [(v, m) for v in vnames for m in [None] + list(range(8))]
    probs = np.zeros((len(names), len(ids), len(keys), 5, 3)); vis = np.zeros((len(ids), len(keys), 5), dtype=bool)
    for i, sid in enumerate(ids):
        clock.check(); P = E2.Patient(DATA, rows[sid], sid); evs = E2.eval_patient(P, prefix)
        for (mm, kk) in evs.values(): require(bool(torch.isfinite(mm).all()), 'Nonfinite view ' + sid)
        kidx = list(range(len(keys)))
        for m_i, (net, pri) in enumerate(models):
            out = R.predict(net, kidx, lambda q: evs[keys[q]], pri, mb=11)
            for q in kidx: probs[m_i, i, q] = out[q][0]; vis[i, q] = out[q][1]
        if (i + 1) % 50 == 0: ev('predict', n=i + 1)
    np.savez(OUT / 'PREDICTIONS.npz', probs=probs, visible=vis, models=np.array(names), patients=np.array(ids), keys=np.array(['%s|%s' % k for k in keys]))
    save(OUT / 'PREDICTIONS_LOCK.json', {'set': which, 'role': role, 'predictions_sha256': sha(OUT / 'PREDICTIONS.npz'), 'labels_read_before_lock': False})
    ev('predictions_locked')
    return names, ids, vnames, keys, probs, vis


def auroc_boot(score, err, pid, n_pat, np):
    """Patient-bootstrap AUROC (errors positive) with ties = 0.5; returns point and percentile CI."""
    order = np.argsort(score, kind='mergesort'); s = score[order]; y = err[order].astype(float); p = pid[order]
    grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])])

    def auc(w):
        wp = w * y; wn = w * (1 - y); W1, W0 = wp.sum(), wn.sum()
        if W1 == 0 or W0 == 0: return np.nan
        gn = np.bincount(grp, weights=wn); below = np.concatenate([[0], np.cumsum(gn)[:-1]])
        return float((wp * (below[grp] + 0.5 * gn[grp])).sum() / (W1 * W0))
    point = auc(np.ones_like(y)); rng = np.random.default_rng(BSEED); vals = []
    for _ in range(B):
        mult = np.bincount(rng.integers(0, n_pat, n_pat), minlength=n_pat); vals.append(auc(mult[p].astype(float)))
    vals = np.array(vals); vals = vals[np.isfinite(vals)]
    return {'auroc': point, 'ci95': [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))], 'levels': int(len(y)), 'errors': int(y.sum())}


def phase_b(which, names, ids, vnames, keys, probs, vis, np):
    from cs_metrics import grade, js_bits
    labels = labels_from_train_csv(ids); Y = np.array([labels[s] for s in ids])
    ki = {k: i for i, k in enumerate(keys)}; n = len(ids)
    ens = {sc: probs[[i for i, nm in enumerate(names) if nm.startswith(sc + '_')]].mean(0) for sc in SCHEMES}

    def sysp(name):
        """[n, 11 views, 5, 3] probabilities for a system."""
        if name.startswith('TTA-'):
            e = ens[name[4:]]; return np.stack([np.mean([e[:, ki[(v, m)]] for m in range(8)], 0) for v in vnames], 1)
        e = ens[name]; return np.stack([e[:, ki[(v, None)]] for v in vnames], 1)
    v0 = vis[:, ki[('I', None)]]; comp = v0.any(-1); vi = {v: i for i, v in enumerate(vnames)}

    def flip(p0, pv):
        f = (grade(p0) != grade(pv)).astype(float); c = v0.sum(-1)
        return np.where(c > 0, (f * v0).sum(-1) / np.maximum(c, 1), np.nan)
    systems = list(SCHEMES) + ['TTA-STD', 'TTA-AUGMIX']; Pm = {s: sysp(s) for s in systems}
    held = {}; t1 = {}
    for s in systems:
        P = Pm[s]; fam = [np.mean([flip(P[:, vi['I']], P[:, vi[v]]) for v in vnames if v.startswith(f + '_')], 0) for f in ('N', 'W', 'D')]
        held[s] = np.mean(fam, 0)[comp]; t1[s] = ((flip(P[:, vi['I']], P[:, vi['T_1_col']]) + flip(P[:, vi['I']], P[:, vi['T_1_row']])) / 2)[comp]
    known = (Y >= 0).any(-1); sev_pat = (Y == 2).any(-1)
    rps = {s: np.array([rps_patient(Y[i], Pm[s][i, vi['I']]) for i in range(n) if known[i]]) for s in systems}
    sevm = {s: np.array([float(severe_miss(Y[i], Pm[s][i, vi['I']])) for i in range(n) if sev_pat[i]]) for s in systems}
    rng = np.random.default_rng(BSEED); idxF = rng.integers(0, int(comp.sum()), (B, int(comp.sum())))
    idxR = np.random.default_rng(BSEED).integers(0, int(known.sum()), (B, int(known.sum())))
    idxS = np.random.default_rng(BSEED).integers(0, max(1, int(sev_pat.sum())), (B, max(1, int(sev_pat.sum()))))

    def d(a, b, idx):
        x = a - b; m = x[idx].mean(1); return {'mean_delta': float(x.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}

    def mci(x, idx):
        m = x[idx].mean(1); return {'mean': float(x.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}
    res = {'set': which, 'patients': n, 'flip_patients': int(comp.sum()), 'rps_patients': int(known.sum()), 'severe_label_patients': int(sev_pat.sum()),
           'heldout_flip': {s: mci(held[s], idxF) for s in systems}, 'T1px_flip': {s: mci(t1[s], idxF) for s in systems},
           'RPS': {s: float(rps[s].mean()) for s in systems}, 'severe_miss_proportion': {s: float(sevm[s].mean()) if len(sevm[s]) else None for s in systems},
           'severe_miss_patients': {s: int(sevm[s].sum()) for s in systems}}
    R1 = {'AUGMIX-STD_flip': d(held['AUGMIX'], held['STD'], idxF), 'AUGMIX-AUGCE_flip': d(held['AUGMIX'], held['AUGCE'], idxF),
          'AUGMIX-STD_RPS': d(rps['AUGMIX'], rps['STD'], idxR), 'AUGMIX-STD_severe_prop': d(sevm['AUGMIX'], sevm['STD'], idxS)}
    R1['conditions'] = {'flip_vs_STD': R1['AUGMIX-STD_flip']['ci95'][1] < 0, 'flip_vs_AUGCE': R1['AUGMIX-AUGCE_flip']['ci95'][1] < 0,
                        'RPS_noninferior_0.003': R1['AUGMIX-STD_RPS']['ci95'][1] <= .003, 'severe_noninferior_0.02': R1['AUGMIX-STD_severe_prop']['ci95'][1] <= .02}
    R1['all_conditions'] = bool(all(R1['conditions'].values()))
    R2 = {'TTA-STD_minus_STD_flip': d(held['TTA-STD'], held['STD'], idxF), 'RPS': d(rps['TTA-STD'], rps['STD'], idxR), 'severe_prop': d(sevm['TTA-STD'], sevm['STD'], idxS)}
    R2['conditions'] = {'flip': R2['TTA-STD_minus_STD_flip']['ci95'][1] < 0, 'RPS_noninferior_0.003': R2['RPS']['ci95'][1] <= .003, 'severe_noninferior_0.02': R2['severe_prop']['ci95'][1] <= .02}
    R2['passed'] = bool(all(R2['conditions'].values()))
    # R3: STD ensemble sensitivity (8 TTA jitters vs original) as a level-level misgrading detector
    e = ens['STD']; pI = e[:, ki[('I', None)]]; sens = np.mean([js_bits(pI, e[:, ki[('I', m)]]) for m in range(8)], 0)
    ok = v0 & (Y >= 0); err = (grade(pI) != Y) & ok; pid = np.repeat(np.arange(n)[:, None], 5, 1)
    R3 = {'sensitivity': auroc_boot(sens[ok], err[ok], pid[ok], n, np), 'one_minus_maxprob': auroc_boot(1 - pI.max(-1)[ok], err[ok], pid[ok], n, np)}
    R3['passed'] = bool(R3['sensitivity']['ci95'][0] > .5)
    collapse = {s: {'predicted_class_distribution': [int((Pm[s][:, vi['I']].argmax(-1) == c).sum()) for c in range(3)], 'mean_max_prob': float(Pm[s][:, vi['I']].max(-1).mean())} for s in systems}
    recall = {s: {str(c): (float(np.mean([Pm[s][i, vi['I'], j].argmax() == c for i in range(n) for j in range(5) if Y[i, j] == c])) if (Y == c).any() else None) for c in range(3)} for s in systems}
    res.update(R1=R1, R2=R2, R3=R3, collapse_indicators=collapse, recall_by_grade=recall)
    return res


def main():
    a = argparse.ArgumentParser(); a.add_argument('--set', choices=['selection', 'policy'], required=True); which = a.parse_args().set
    clock = Clock(5900 if which == 'policy' else 3500); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'CS2_eval_' + which}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for nm, h in inv['files_sha256'].items(): require(sha(HERE / nm) == h, 'Inventory mismatch ' + nm)
        dev = None
        if which == 'policy':
            dev = load(CS / 'v2evalsel/out/EVAL_RESULT.json'); require(dev['status'] == 'completed', 'Development evaluation missing')
        import numpy as np, torch
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        with GpuLock('CS2_eval_' + which):
            A = phase_a(which, clock, ev, np, torch)
        result.update(predictions_lock_sha256=sha(OUT / 'PREDICTIONS_LOCK.json'))
        result.update(phase_b(which, *A, np))
        if which == 'policy':
            dev_ok = dev['R1']['AUGMIX-STD_flip']['mean_delta'] < 0
            result['R1']['development_gate_point_estimate_lt_0'] = bool(dev_ok)
            result['R1']['passed'] = bool(dev_ok and result['R1']['all_conditions'])
            result['R1']['status'] = 'judged' if dev_ok else 'development_not_met_not_confirmed'
        result['status'] = 'completed'; ev('done')
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'EVAL_RESULT.json', result)


if __name__ == '__main__':
    main()
