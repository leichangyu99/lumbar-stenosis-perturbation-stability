"""CS20261013-MA evaluation: 27 frozen evaluation views generated on the fly for confirmation 197 and
policy474; 12 MA models (4 archs x 3 seeds) predict; reference grades were completed before predictions.
Primary (label-free): per architecture and set, T 1px flip CI lower > 0; secondary dose JS(1.0)-JS(0.25).
Descriptive: family flips, Kendall W of family ranking across architectures, RPS/recall (labels previously read)."""
import argparse, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, DATA, PRED, SEEDS, Clock, GpuLock, labels_from_train_csv, load, require, rps_patient, save, severe_miss, sha  # noqa: E402

WEIGHTS = CS / 'ma/weights'
ARCH_LIST = ['A0', 'A1', 'A2', 'A3']
TAGS = ['A', 'B']
B, BSEED = 10000, 20261013


def predict_set(role, models, priors, clock, ev, np, torch):
    import cs_engine as E, cs2_engine as E2, cs3_arch as A
    ids = load(CS / 'e0/strategy_split.json')['patient_ids'][role]
    if role == 'source_confirmation':
        rows = {r['study_id']: r for r in load(CS / 'e0/CONFIRMATION_LOCALIZATION.json')['predictions']}
    else:
        rows = {r['study_id']: r for r in load(PRED)['predictions'] if r['role'] == role}
    require(set(rows) >= set(ids), 'rows missing ' + role)
    views = E.evaluation_views(); vn = [v['name'] for v in views]
    probs = np.zeros((len(models), len(ids), len(vn), 5, 3)); vis = np.zeros((len(ids), len(vn), 5), dtype=bool)
    for i, sid in enumerate(ids):
        clock.check(); pv = A.patient_views(E2.Patient(DATA, rows[sid], sid), views)
        X = torch.stack([pv[n][0] for n in vn]).cuda(); M = torch.stack([pv[n][1] for n in vn]).cuda(); V = M.any(-1)
        vis[i] = V.cpu().numpy()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for m_i, (name, net) in enumerate(models):
                z = torch.cat([net(X[s:s + 9], M[s:s + 9]).float() for s in range(0, len(vn), 9)]).softmax(-1).double()
                z = torch.where(V[..., None], z, priors[name].double().cuda()[None].expand(len(vn), -1, -1))
                require(bool(torch.isfinite(z).all()), 'Nonfinite ' + sid); probs[m_i, i] = z.cpu().numpy()
        if (i + 1) % 50 == 0: ev('predict', role=role, n=i + 1)
    return ids, vn, probs, vis


def stats(role, names, ids, vn, probs, vis, np):
    from cs_metrics import grade, js_bits
    vi = {v: i for i, v in enumerate(vn)}; v0 = vis[:, vi['I']]; comp = v0.any(-1)
    idx = np.random.default_rng(BSEED).integers(0, int(comp.sum()), (B, int(comp.sum())))

    def ci(x): m = x[idx].mean(1); return {'mean': float(x.mean()), 'ci95': [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]}

    def fj(p0, pv):
        f = (grade(p0) != grade(pv)).astype(float); j = js_bits(p0, pv); c = v0.sum(-1)
        return (f * v0).sum(-1)[comp] / c[comp], (j * v0).sum(-1)[comp] / c[comp]
    out = {'patients': len(ids), 'flip_patients': int(comp.sum())}; fam_means = {}
    for arch in ARCH_LIST:
        P = probs[[k for k, n in enumerate(names) if n.startswith(arch + '_')]].mean(0)
        pv = {v: fj(P[:, vi['I']], P[:, vi[v]]) for v in vn}
        t1 = (pv['T_1_col'][0] + pv['T_1_row'][0]) / 2; dj = (pv['T_1_col'][1] + pv['T_1_row'][1]) / 2 - (pv['T_0.25_col'][1] + pv['T_0.25_row'][1]) / 2
        fam = {f: float(np.mean([pv[v][0].mean() for v in vn if v.startswith(f + '_')])) for f in ('T', 'R', 'S', 'N', 'W', 'D')}; fam_means[arch] = fam
        r = {'identity_zero': bool(max(pv['I'][0].max(), pv['I'][1].max()) == 0), 'T1px_flip': ci(t1), 'dose_JS1_minus_JS025': ci(dj),
             'family_flip_means': fam, 'per_view_flip': {v: float(pv[v][0].mean()) for v in vn if v != 'I'}}
        r['primary_passed'] = bool(r['identity_zero'] and r['T1px_flip']['ci95'][0] > 0); r['dose_trend'] = bool(r['dose_JS1_minus_JS025']['ci95'][0] > 0)
        per_seed = {}
        for k, n in enumerate(names):
            if n.startswith(arch + '_'):
                Q = probs[k]; per_seed[n] = float(((fj(Q[:, vi['I']], Q[:, vi['T_1_col']])[0] + fj(Q[:, vi['I']], Q[:, vi['T_1_row']])[0]) / 2).mean())
        r['T1px_flip_per_seed'] = per_seed; out[arch] = r
    # Kendall W of family flip ranking across architectures
    from scipy.stats import rankdata
    R = np.array([rankdata([fam_means[a][f] for f in ('T', 'R', 'S', 'N', 'W', 'D')]) for a in ARCH_LIST]); m, n = R.shape
    S = ((R.sum(0) - m * (n + 1) / 2) ** 2).sum(); out['kendall_W_family_ranking'] = float(12 * S / (m ** 2 * (n ** 3 - n)))
    return out


def quality(role, names, ids, vn, probs, np):
    labels = labels_from_train_csv(ids); Y = np.array([labels[s] for s in ids]); vi = vn.index('I'); res = {}
    for arch in ARCH_LIST:
        P = probs[[k for k, n in enumerate(names) if n.startswith(arch + '_')]].mean(0)[:, vi]
        rp = [rps_patient(Y[i], P[i]) for i in range(len(ids))]; known = [i for i in range(len(ids)) if (Y[i] >= 0).any()]
        rec = {str(c): (float(np.mean([P[i, j].argmax() == c for i in range(len(ids)) for j in range(5) if Y[i, j] == c])) if (Y == c).any() else None) for c in range(3)}
        res[arch] = {'RPS': float(np.mean([x for x in rp if x is not None])), 'severe_miss_patients': int(sum(severe_miss(Y[i], P[i]) for i in known)), 'recall_by_grade': rec}
    return res


def main():
    a = argparse.ArgumentParser(); a.add_argument('--archs', nargs='+'); a.add_argument('--tags', nargs='+'); args = a.parse_args()
    if args.archs: ARCH_LIST[:] = args.archs
    if args.tags: TAGS[:] = args.tags
    clock = Clock(5900); out = HERE / 'out'; out.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'MA_eval'}
    log = (out / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        import numpy as np, torch
        import cs3_arch as A
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        fits = {}
        for tag in TAGS:
            tr = load(CS / ('matr' + tag) / ('out_' + tag) / 'TRAIN_RESULT.json'); require(tr['status'] == 'completed', 'training ' + tag); fits.update({k: (tag, v) for k, v in tr['fits'].items()})
        with GpuLock('MA_eval'):
            models = []; priors = {}
            for arch in ARCH_LIST:
                for seed in SEEDS:
                    n = '%s_%d' % (arch, seed); tag, f = fits[n]; p = CS / ('matr' + tag) / ('out_' + tag) / n / 'FINAL_EPOCH12.pt'
                    require(sha(p) == f['checkpoint_sha256'], 'Checkpoint changed ' + n); ck = torch.load(p, map_location='cpu', weights_only=True)
                    net = A.Grader(arch, WEIGHTS / A.weight_file(arch)).cuda(); net.load_state_dict(ck['state_dict']); net.eval()
                    models.append((n, net)); priors[n] = ck['priors']
            names = [n for n, _ in models]; sets = {}
            for role, tag in (('source_confirmation', 'confirmation'), ('policy_train', 'policy')):
                ids, vn, probs, vis = predict_set(role, models, priors, clock, ev, np, torch)
                np.savez(out / ('PRED_%s.npz' % tag), probs=probs, visible=vis, models=np.array(names), patients=np.array(ids), views=np.array(vn))
                sets[tag] = (ids, vn, probs, vis)
        save(out / 'PREDICTIONS_LOCK.json', {k: sha(out / ('PRED_%s.npz' % k)) for k in sets}); ev('locked')
        result['flip'] = {k: stats(k, names, *v, np) for k, v in sets.items()}
        result['quality_descriptive'] = {k: quality(k, names, v[0], v[1], v[2], np) for k, v in sets.items()}
        result['dev_RPS_model_selection'] = {n: fits[n][1]['dev_RPS_model_selection'] for n in names}
        dev = {a: float(np.mean([fits['%s_%d' % (a, s)][1]['dev_RPS_model_selection'] for s in SEEDS])) for a in ARCH_LIST}
        if 'A0' not in dev:   # A0 reference trained in matrA (fixed after maevalC bookkeeping failure)
            ref = load(CS / 'matrA/out_A/TRAIN_RESULT.json')['fits']; dev['A0'] = float(np.mean([ref['A0_%d' % s]['dev_RPS_model_selection'] for s in SEEDS]))
        result['quality_flag_dev_RPS_worse_than_A0_by_0.01'] = {a: bool(dev[a] > dev['A0'] + .01) for a in ARCH_LIST}
        passes = {f'{a}_{k}': result['flip'][k][a]['primary_passed'] for a in ARCH_LIST for k in sets}
        result.update(primary_tests=passes, generality_all8_passed=bool(all(passes.values())), status='completed'); ev('done')
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(out / 'MA_RESULT.json', result)


if __name__ == '__main__':
    main()
