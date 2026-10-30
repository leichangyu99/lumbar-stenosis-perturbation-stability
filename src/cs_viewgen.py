"""CS20261012 view generation (EXPERIMENT_PLAN V3 section 2, budget block 'view generation').

- confirmation (197): 27 evaluation views; identity generated twice and required byte-identical.
- diagnostic_train (1107): fixed training view V (identity views are the verified front_cache).
- development stress test (dev20 from subset300): T family at 3x doses {0.75,1.5,3.0} px.
No labels are read.  Outputs: CS/views/{confirmation,train_V,stress}/<sid>.pt + VIEWGEN_RESULT.json.
"""
import hashlib, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, DATA, PRED, WEIGHTS, Clock, GpuLock, hrank, load, require, save, sha  # noqa: E402

OUT = CS / 'views'


def main():
    clock = Clock(3300); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'viewgen', 'labels_read': False}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        import json
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        e0 = load(CS / 'e0/results/E0_REMOTE_RESULT.json'); require(e0['status'] == 'completed' and all(e0['gates'].values()), 'E0 not passed')
        split = load(CS / 'e0/strategy_split.json')['patient_ids']; conf = split['source_confirmation']; diag = split['diagnostic_train']
        loc = load(CS / 'e0/CONFIRMATION_LOCALIZATION.json'); crow = {r['study_id']: r for r in loc['predictions']}; require(set(crow) == set(conf), 'Localisation rows')
        drow = {r['study_id']: r for r in load(PRED)['predictions'] if r['role'] == 'diagnostic_train'}
        dev = e0['dev20']
        import numpy as np, torch
        import cs_engine as E
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        with GpuLock('CS_viewgen'):
            _, prefix = E.load_prefix(WEIGHTS, 'cuda'); views = E.evaluation_views(); ev('start')
            files = {}; conf_stats = {v['name']: {'slices': 0, 'single_slice_levels_not_dropped': 0} for v in views}; repeat_ok = True
            (OUT / 'confirmation').mkdir()
            for i, sid in enumerate(conf):
                clock.check()
                allv = E.build_views(DATA, crow[sid], sid, views, prefix)
                again = E.build_views(DATA, crow[sid], sid, [views[0]], prefix)['I']
                same = hashlib.sha256(allv['I'][0].numpy().tobytes()).hexdigest() == hashlib.sha256(again[0].numpy().tobytes()).hexdigest() and torch.equal(allv['I'][1], again[1])
                repeat_ok &= bool(same)
                for v in views:
                    m, k, info = allv[v['name']]; require(bool(torch.isfinite(m).all()), 'Nonfinite ' + sid)
                    conf_stats[v['name']]['slices'] += int(k.sum()); conf_stats[v['name']]['single_slice_levels_not_dropped'] += info.get('levels_with_single_slice_not_dropped', 0)
                p = OUT / 'confirmation' / (sid + '.pt')
                torch.save({'study_id': sid, 'role': 'source_confirmation', 'views': {n: {'maps': m, 'mask': k} for n, (m, k, _) in allv.items()}}, p)
                files['confirmation/' + sid + '.pt'] = sha(p)
                if (i + 1) % 25 == 0: ev('confirmation', n=i + 1)
            require(repeat_ok, 'Confirmation identity view not byte-identical on repeat')
            (OUT / 'train_V').mkdir(); missing = []; vparams = {}
            for i, sid in enumerate(diag):
                clock.check(); tv = E.train_view(sid); vparams[sid] = {k: tv[k] for k in ('tx', 'ty', 'theta_deg', 's')}
                if sid in drow:
                    m, k, _ = E.build_views(DATA, drow[sid], sid, [tv], prefix)['V']
                else:
                    missing.append(sid); m = torch.zeros(5, 3, 256, 14, 14); k = torch.zeros(5, 3, dtype=torch.bool)
                require(bool(torch.isfinite(m).all()), 'Nonfinite V ' + sid)
                p = OUT / 'train_V' / (sid + '.pt'); torch.save({'study_id': sid, 'role': 'diagnostic_train', 'view': vparams[sid], 'maps': m, 'mask': k}, p)
                files['train_V/' + sid + '.pt'] = sha(p)
                if (i + 1) % 200 == 0: ev('train_V', n=i + 1)
            (OUT / 'stress').mkdir()
            stress_views = [E.view('T3x_%g_%s' % (d, ax), 'T', **{('tx' if ax == 'col' else 'ty'): d}, dose=d) for d in (0.75, 1.5, 3.0) for ax in ('col', 'row')]
            for sid in dev:
                clock.check(); allv = E.build_views(DATA, drow[sid], sid, [views[0]] + stress_views, prefix)
                p = OUT / 'stress' / (sid + '.pt'); torch.save({'study_id': sid, 'role': 'diagnostic_train', 'views': {n: {'maps': m, 'mask': k} for n, (m, k, _) in allv.items()}}, p)
                files['stress/' + sid + '.pt'] = sha(p)
            save(OUT / 'TRAIN_VIEW_PARAMS.json', vparams)
            result.update(status='completed', confirmation_patients=len(conf), confirmation_identity_repeat_byte_identical=bool(repeat_ok), confirmation_view_stats=conf_stats,
                          train_V_patients=len(diag), train_V_missing_sagittal=missing, stress_patients=len(dev), stress_views=[v['name'] for v in stress_views],
                          files_sha256=files, train_view_params_sha256=sha(OUT / 'TRAIN_VIEW_PARAMS.json'))
            ev('done')
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'VIEWGEN_RESULT.json', result)


if __name__ == '__main__':
    main()
