"""CS20261012 E5 (C3b): retrospective sensitivity-guided active learning (EXPERIMENT_PLAN V3 E5).

Pool = diagnostic_train 1107; evaluation = model_selection 197 original view (front_cache).  Per seed:
S0 = first 100 by SHA256('CS20261012|al_start|seed|ID') (shared by the three strategies).  Each round
retrains the STD recipe from ImageNet on acquired patients only (labels, class weights, priors and
feature normalisation from acquired patients only); the unacquired pool is only predicted for query
scores.  RANDOM by SHA256('CS20261012|al_rand|seed|ID'); ENTROPY = mean natural-log entropy over visible
original-view levels; SENS = mean JS(original, V) over levels visible in both; no visible level -> -inf;
ties by SHA256('CS20261012|tie|ID'); +50 per round up to 600 (11 fits per strategy and seed, 99 fits).
AULC = trapezoid over the 11 sizes / 500; patient bootstrap over whole curves.  No checkpoints kept.
"""
import io, json, math, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CACHE, CS, DIAG_LABELS, DIAG_LABELS_SHA, SEEDS, Clock, GpuLock, Recipe, boot_indices, ci, hrank, labels_from_train_csv, load, require, rps_patient, save, sha  # noqa: E402

OUT = HERE / 'out'
SIZES = list(range(100, 601, 50))
STRATS = ('RANDOM', 'ENTROPY', 'SENS')


def main():
    clock = Clock(7000); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'E5'}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        vg = load(CS / 'views/VIEWGEN_RESULT.json'); require(vg['status'] == 'completed', 'Viewgen')
        split = load(CS / 'e0/strategy_split.json')['patient_ids']; pool = split['diagnostic_train']; sel = split['model_selection']
        require(len(pool) == 1107 and len(sel) == 197, 'sizes')
        require(sha(DIAG_LABELS) == DIAG_LABELS_SHA, 'Labels changed'); plabels = load(DIAG_LABELS)['labels']
        import numpy as np, torch
        from cs_metrics import js_bits
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        with GpuLock('CS_E5'):
            gI, gV, gS = {}, {}, {}
            def cache(sid, role):
                rec = torch.load(io.BytesIO((CACHE / (sid + '.pt')).read_bytes()), map_location='cpu', weights_only=True)
                require(rec['study_id'] == sid and rec['role'] == role, 'Cache role ' + sid); return rec['maps'].cuda(), rec['mask'].cuda()
            for sid in pool:
                clock.check(); gI[sid] = cache(sid, 'diagnostic_train')
                vp = CS / 'views/train_V' / (sid + '.pt'); require(sha(vp) == vg['files_sha256']['train_V/' + sid + '.pt'], 'V changed')
                v = torch.load(vp, map_location='cpu', weights_only=True); gV[sid] = (v['maps'].cuda(), v['mask'].cuda())
            for sid in sel: gS[sid] = cache(sid, 'model_selection')
            slabels = labels_from_train_csv(sel); Y = np.array([slabels[s] for s in sel]); known = (Y >= 0).any(-1)
            save(OUT / 'SELECTION_LABELS_READ.json', {'role': 'model_selection', 'patients': len(sel), 'known_patients': int(known.sum()), 'read_at_elapsed_s': clock.elapsed()})
            ev('inputs_resident')
            R = Recipe(torch, np); curves = {}; acq_log = {}
            tie = {s: h for s, h in zip(hrank('tie', pool), range(len(pool)))}
            for seed in SEEDS:
                S0 = hrank('al_start|%d' % seed, pool)[:100]; rand_order = hrank('al_rand|%d' % seed, pool)
                for strat in STRATS:
                    acquired = list(S0); curve = []; rounds = []
                    for size in SIZES:
                        clock.check(); require(len(acquired) == size and len(set(acquired)) == size, 'Acquisition size')
                        net, priors, info = R.fit(acquired, plabels, gI, gV, 'STD', seed, clock.check)
                        pe = R.predict(net, sel, lambda s: gS[s], priors)
                        rp = [rps_patient(Y[i], pe[s][0]) for i, s in enumerate(sel)]
                        curve.append([x if x is not None else float('nan') for x in rp])
                        rounds.append({'size': size, 'mean_RPS': float(np.nanmean([x for x in rp if x is not None])), 'class_counts': info['class_counts']})
                        if size < SIZES[-1]:
                            aset = set(acquired); rest = [s for s in pool if s not in aset]
                            if strat == 'RANDOM':
                                new = [s for s in rand_order if s not in aset][:50]
                            else:
                                pI = R.predict(net, rest, lambda s: gI[s], priors); score = {}
                                if strat == 'SENS': pV = R.predict(net, rest, lambda s: gV[s], priors)
                                for s in rest:
                                    p, v = pI[s]
                                    if strat == 'ENTROPY':
                                        h = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(-1)
                                        score[s] = float(h[v].mean()) if v.any() else -math.inf
                                    else:
                                        both = v & pV[s][1]; j = js_bits(p, pV[s][0])
                                        score[s] = float(j[both].mean()) if both.any() else -math.inf
                                new = sorted(rest, key=lambda s: (-score[s], tie[s]))[:50]
                            acquired += new
                        del net; torch.cuda.empty_cache()
                        ev('round', seed=seed, strategy=strat, size=size, mean_RPS=rounds[-1]['mean_RPS'])
                    curves['%s_%d' % (strat, seed)] = curve; acq_log['%s_%d' % (strat, seed)] = {'order': acquired, 'rounds': rounds}
                    save(OUT / 'ACQUISITIONS.json', acq_log)
            # ---- AULC and C3b ----
            idx_known = np.where(known)[0]
            def aulc(key):
                c = np.array(curves[key])[:, idx_known]     # [11, n]
                return np.trapezoid(c, x=np.array(SIZES), axis=0) / 500.0
            A = {s: np.mean([aulc('%s_%d' % (s, seed)) for seed in SEEDS], 0) for s in STRATS}
            idx = boot_indices(len(idx_known))
            comps = {'SENS-RANDOM': {'mean_delta': float((A['SENS'] - A['RANDOM']).mean()), 'ci95': ci(A['SENS'] - A['RANDOM'], idx)},
                     'SENS-ENTROPY': {'mean_delta': float((A['SENS'] - A['ENTROPY']).mean()), 'ci95': ci(A['SENS'] - A['ENTROPY'], idx)}}
            mean_curves = {s: [float(np.mean([np.nanmean(np.array(curves['%s_%d' % (s, seed)])[k, idx_known]) for seed in SEEDS])) for k in range(len(SIZES))] for s in STRATS}
            result.update(status='completed', fits=len(SIZES) * len(STRATS) * len(SEEDS), AULC={s: float(A[s].mean()) for s in STRATS}, comparisons=comps,
                          C3b_passed=bool(comps['SENS-RANDOM']['ci95'][1] < 0 and comps['SENS-ENTROPY']['ci95'][1] < 0), mean_curves=mean_curves, sizes=SIZES,
                          evaluation_patients=int(len(idx_known)), checkpoints_kept=False)
            save(OUT / 'CURVES.json', curves)
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'E5_RESULT.json', result)


if __name__ == '__main__':
    main()
