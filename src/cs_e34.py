"""CS20261012 E3 (C1 input-intervention sensitivity) and E4 (C3a robust learning) on the confirmation set.

Phase A (no labels): all 12 frozen E2 models predict the 27 confirmation views; STD models also predict the
dev20 stress views (T at 3x dose).  Predictions are written and hashed into PREDICTIONS_LOCK.json.
Phase B: confirmation labels are read once from train.csv (source_confirmation rows only) and the frozen
statistics of EXPERIMENT_PLAN V3 E3/E4 are computed (patient bootstrap 10000, seed 20261012, percentile CI).
"""
import json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, SEEDS, Clock, GpuLock, Recipe, boot_indices, ci, labels_from_train_csv, load, require, rps_patient, save, severe_miss, sha  # noqa: E402

OUT = HERE / 'out'
MODES = ('STD', 'AUG', 'CONS', 'SHUF')


def phase_a(clock, ev, np, torch):
    vg = load(CS / 'views/VIEWGEN_RESULT.json'); require(vg['status'] == 'completed' and vg['confirmation_identity_repeat_byte_identical'], 'Viewgen')
    e2 = load(CS / 'e2/out/E2_RESULT.json'); require(e2['status'] == 'completed' and len(e2['fits']) == 12, 'E2 incomplete')
    split = load(CS / 'e0/strategy_split.json')['patient_ids']; conf = split['source_confirmation']
    import cs_engine as E
    vnames = [v['name'] for v in E.evaluation_views()]
    R = Recipe(torch, np); models = {}
    for mode in MODES:
        for seed in SEEDS:
            name = '%s_%d' % (mode, seed); p = CS / 'e2/out' / name / 'FINAL_EPOCH12.pt'
            require(sha(p) == e2['fits'][name]['checkpoint_sha256'], 'Checkpoint changed ' + name)
            ck = torch.load(p, map_location='cpu', weights_only=True)
            net = R.Diagnosis(R.backbone.layer4, R.SagHead().cuda(), torch.zeros(512).cuda(), torch.ones(512).cuda(), True).cuda()
            net.load_state_dict(ck['state_dict']); net.eval(); models[name] = (net, ck['priors'])
    names = list(models)
    probs = np.zeros((len(names), len(conf), len(vnames), 5, 3)); vis = np.zeros((len(conf), len(vnames), 5), dtype=bool)
    for i, sid in enumerate(conf):
        clock.check(); fp = CS / 'views/confirmation' / (sid + '.pt'); require(sha(fp) == vg['files_sha256']['confirmation/' + sid + '.pt'], 'View file changed ' + sid)
        rec = torch.load(fp, map_location='cpu', weights_only=True); require(rec['study_id'] == sid, 'id')
        for m_i, n in enumerate(names):
            net, pri = models[n]
            out = R.predict(net, vnames, lambda v: (rec['views'][v]['maps'], rec['views'][v]['mask']), pri)
            for v_i, v in enumerate(vnames):
                probs[m_i, i, v_i] = out[v][0]; vis[i, v_i] = out[v][1]
        if (i + 1) % 50 == 0: ev('predict_confirmation', n=i + 1)
    # stress test (development, STD models only)
    stress = {}
    sdir = CS / 'views/stress'
    for fp in sorted(sdir.glob('*.pt')):
        sid = fp.stem; require(sha(fp) == vg['files_sha256']['stress/' + fp.name], 'Stress file changed')
        rec = torch.load(fp, map_location='cpu', weights_only=True); sv = list(rec['views'])
        stress[sid] = {}
        for n in names:
            if not n.startswith('STD_'): continue
            net, pri = models[n]; out = R.predict(net, sv, lambda v: (rec['views'][v]['maps'], rec['views'][v]['mask']), pri)
            stress[sid][n] = {v: (out[v][0].tolist(), out[v][1].tolist()) for v in sv}
    np.savez(OUT / 'PREDICTIONS.npz', probs=probs, visible=vis, models=np.array(names), patients=np.array(conf), views=np.array(vnames))
    save(OUT / 'STRESS_PREDICTIONS.json', stress)
    lock = {'predictions_sha256': sha(OUT / 'PREDICTIONS.npz'), 'stress_sha256': sha(OUT / 'STRESS_PREDICTIONS.json'), 'e2_result_sha256': sha(CS / 'e2/out/E2_RESULT.json'),
            'viewgen_result_sha256': sha(CS / 'views/VIEWGEN_RESULT.json'), 'labels_read_before_lock': False}
    save(OUT / 'PREDICTIONS_LOCK.json', lock); ev('predictions_locked')
    return names, conf, vnames, probs, vis, stress


def flips_js(p0, pv, v0, np):
    """Per-patient flip proportion and JS/ln2 mean over levels visible in the ORIGINAL view; NaN if none."""
    from cs_metrics import grade, js_bits
    f = (grade(p0) != grade(pv)).astype(float); j = js_bits(p0, pv)
    cnt = v0.sum(-1); fl = np.where(cnt > 0, (f * v0).sum(-1) / np.maximum(cnt, 1), np.nan); jj = np.where(cnt > 0, (j * v0).sum(-1) / np.maximum(cnt, 1), np.nan)
    return fl, jj


def phase_b(names, conf, vnames, probs, vis, stress, np):
    labels = labels_from_train_csv(conf); Y = np.array([labels[s] for s in conf])
    vi = {v: i for i, v in enumerate(vnames)}; ens = {m: probs[[k for k, n in enumerate(names) if n.startswith(m + '_')]].mean(0) for m in MODES}
    v0 = vis[:, vi['I'], :]; comp = v0.any(-1); nonc = int((~comp).sum())
    idxF = boot_indices(int(comp.sum())); B = idxF.shape[0]

    def summ(x, idx):
        x = np.asarray(x)[comp] if len(x) == len(conf) else np.asarray(x)
        return {'mean': float(x.mean()), 'ci95': ci(x, idx)}

    def boot_p(x, idx):
        m = np.asarray(x)[idx].mean(1); return float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
    res = {'labels_read': 'source_confirmation only, once, after PREDICTIONS_LOCK', 'patients': len(conf), 'flip_computable_patients': int(comp.sum()), 'flip_noncomputable_patients': nonc}
    # identity must be exactly 0
    fI, jI = flips_js(ens['STD'][:, vi['I']], ens['STD'][:, vi['I']], v0, np); res['identity_flip_js_zero'] = bool(np.nanmax(fI) == 0 and np.nanmax(jI) == 0)
    # ---- E3 / C1 (STD ensemble) ----
    pS = ens['STD']; per_view = {}
    for v in vnames:
        f, j = flips_js(pS[:, vi['I']], pS[:, vi[v]], v0, np); per_view[v] = (f[comp], j[comp])
    t1 = (per_view['T_1_col'][0] + per_view['T_1_row'][0]) / 2
    js1 = (per_view['T_1_col'][1] + per_view['T_1_row'][1]) / 2; js025 = (per_view['T_0.25_col'][1] + per_view['T_0.25_row'][1]) / 2
    c1 = {'T1px_flip': summ(t1, idxF), 'dose_JS1_minus_JS025': summ(js1 - js025, idxF)}
    c1['C1_passed'] = bool(res['identity_flip_js_zero'] and comp.sum() > 0 and c1['T1px_flip']['ci95'][0] > 0)
    c1['dose_trend_confirmed'] = bool(c1['dose_JS1_minus_JS025']['ci95'][0] > 0)
    desc = {}; pv_raw = {}
    for v in vnames:
        if v == 'I': continue
        f, j = per_view[v]; desc[v] = {'flip': summ(f, idxF), 'js': summ(j, idxF), 'flip_patients_any': int((f > 0).sum())}; pv_raw[v] = boot_p(f, idxF)
    order = sorted(pv_raw, key=pv_raw.get); m = len(order); running = 0.
    for r, v in enumerate(order):
        running = max(running, min(1.0, (m - r) * pv_raw[v])); desc[v]['flip_boot_sign_p'] = pv_raw[v]; desc[v]['flip_holm_p'] = running
    fam = {}
    for famname in ('T', 'R', 'S', 'N', 'W', 'D'):
        vs = [v for v in vnames if v.startswith(famname + '_')]
        fam[famname] = {'flip': summ(np.mean([per_view[v][0] for v in vs], 0), idxF), 'js': summ(np.mean([per_view[v][1] for v in vs], 0), idxF), 'views': vs}
    res['E3'] = {'C1': c1, 'per_view_descriptive': desc, 'per_family_descriptive': fam}
    # per-seed descriptive for T1px
    res['E3']['T1px_flip_per_seed'] = {}
    for k, n in enumerate(names):
        if not n.startswith('STD_'): continue
        a, _ = flips_js(probs[k][:, vi['I']], probs[k][:, vi['T_1_col']], v0, np); b, _ = flips_js(probs[k][:, vi['I']], probs[k][:, vi['T_1_row']], v0, np)
        res['E3']['T1px_flip_per_seed'][n] = float(np.mean(((a + b) / 2)[comp]))
    # stress (development, descriptive)
    st = {}
    for sid, d in stress.items():
        for n, vv in d.items():
            p0 = np.array(vv['I'][0]); vz = np.array(vv['I'][1])
            for v, (p, _) in vv.items():
                if v == 'I': continue
                f, j = flips_js(p0[None], np.array(p)[None], vz[None], np)
                st.setdefault(v, []).append((float(f[0]), float(j[0])))
    res['E3']['stress_dev20_STD_per_seed_mean'] = {v: {'flip': float(np.nanmean([x[0] for x in xs])), 'js': float(np.nanmean([x[1] for x in xs]))} for v, xs in st.items()}
    # ---- E4 / C3a ----
    held = {}
    for mname in MODES:
        p = ens[mname]; famv = []
        for famname in ('N', 'W', 'D'):
            vs = [v for v in vnames if v.startswith(famname + '_')]
            famv.append(np.mean([flips_js(p[:, vi['I']], p[:, vi[v]], v0, np)[0][comp] for v in vs], 0))
        held[mname] = np.mean(famv, 0)
    known = (Y >= 0).any(-1); idxR = boot_indices(int(known.sum()))
    rpsv = {mname: np.array([rps_patient(Y[i], ens[mname][i, vi['I']]) for i in range(len(conf)) if known[i]]) for mname in MODES}
    sev = {mname: int(sum(severe_miss(Y[i], ens[mname][i, vi['I']]) for i in range(len(conf)) if known[i])) for mname in MODES}
    def delta(a, b, idx): d = a - b; return {'mean_delta': float(d.mean()), 'ci95': ci(d, idx)}
    comps = {'CONS-AUG': delta(held['CONS'], held['AUG'], idxF), 'CONS-STD': delta(held['CONS'], held['STD'], idxF), 'CONS-SHUF': delta(held['CONS'], held['SHUF'], idxF)}
    rps_d = delta(rpsv['CONS'], rpsv['STD'], idxR)
    conds = {'1_CONS_lt_AUG': comps['CONS-AUG']['ci95'][1] < 0, '2_CONS_lt_STD': comps['CONS-STD']['ci95'][1] < 0, '3_CONS_lt_SHUF': comps['CONS-SHUF']['ci95'][1] < 0,
             '4_RPS_CONS_minus_STD_upper_le_0.003': rps_d['ci95'][1] <= 0.003, '5_severe_CONS_le_STD': sev['CONS'] <= sev['STD']}
    collapse = {}
    for mname in MODES:
        p = ens[mname][:, vi['I']]; g = p.argmax(-1)
        collapse[mname] = {'predicted_class_distribution': [int((g == c).sum()) for c in range(3)], 'mean_max_prob': float(p.max(-1).mean())}
    recall = {mname: {str(c): (float(np.mean([ens[mname][i, vi['I'], j].argmax() == c for i in range(len(conf)) for j in range(5) if Y[i, j] == c])) if (Y == c).any() else None) for c in range(3)} for mname in MODES}
    res['E4'] = {'heldout_flip': {mname: summ(held[mname], idxF) for mname in MODES}, 'comparisons': comps, 'RPS_original': {mname: {'mean': float(rpsv[mname].mean())} for mname in MODES},
                 'RPS_CONS_minus_STD': rps_d, 'severe_patients_with_miss': sev, 'recall_by_grade': recall, 'collapse_indicators': collapse, 'conditions': {k: bool(v) for k, v in conds.items()},
                 'C3a_passed': bool(all(conds.values())), 'rps_patients': int(known.sum())}
    res['bootstrap'] = {'B': B, 'seed': 20261012, 'flip_indices_n': int(comp.sum()), 'rps_indices_n': int(known.sum())}
    return res


def main():
    clock = Clock(3300); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'E3_E4'}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        import numpy as np, torch
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        with GpuLock('CS_E34'):
            a = phase_a(clock, ev, np, torch)
        result.update(predictions_lock_sha256=sha(OUT / 'PREDICTIONS_LOCK.json'))
        result.update(phase_b(*a, np)); result['status'] = 'completed'; ev('done')
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'E3_E4_RESULT.json', result)


if __name__ == '__main__':
    main()
