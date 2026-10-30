"""CS20261012 E2: 12 grader fits (STD/AUG/CONS/SHUF x seeds 20261012/3/4) on diagnostic_train 1107.

Identity views = verified front_cache (E0 bitwise gate); training view V = views/train_V.  Labels are the
diagnostic-only label file (historically touched).  Order follows the stop priority: STD first, SHUF last.
Fixed final epoch 12, no selection.  Outputs: CS/e2/out/<mode>_<seed>/{FINAL_EPOCH12.pt,FIT.json}.
"""
import io, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CACHE, CS, DIAG_LABELS, DIAG_LABELS_SHA, SEEDS, Clock, GpuLock, Recipe, load, require, save, sha  # noqa: E402

OUT = HERE / 'out'
MODES = ('STD', 'AUG', 'CONS', 'SHUF')


def main():
    clock = Clock(5200); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'E2'}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        vg = load(CS / 'views/VIEWGEN_RESULT.json'); require(vg['status'] == 'completed', 'Viewgen not completed')
        split = load(CS / 'e0/strategy_split.json')['patient_ids']; diag = split['diagnostic_train']; require(len(diag) == 1107, 'diag size')
        require(sha(DIAG_LABELS) == DIAG_LABELS_SHA, 'Labels changed'); labels = load(DIAG_LABELS)['labels']; require(set(labels) == set(diag), 'Non-diagnostic labels')
        import numpy as np, torch
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        with GpuLock('CS_E2'):
            gI, gV, cache_sha = {}, {}, {}
            for i, sid in enumerate(diag):
                clock.check()
                raw = (CACHE / (sid + '.pt')).read_bytes(); cache_sha[sid] = __import__('hashlib').sha256(raw).hexdigest()
                rec = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
                require(rec['study_id'] == sid and rec['role'] == 'diagnostic_train' and tuple(rec['maps'].shape) == (5, 3, 256, 14, 14) and bool(torch.isfinite(rec['maps']).all()), 'Cache record ' + sid)
                vp = CS / 'views/train_V' / (sid + '.pt'); require(sha(vp) == vg['files_sha256']['train_V/' + sid + '.pt'], 'V file changed ' + sid)
                v = torch.load(vp, map_location='cpu', weights_only=True); require(v['study_id'] == sid, 'V id')
                gI[sid] = (rec['maps'].cuda(), rec['mask'].cuda()); gV[sid] = (v['maps'].cuda(), v['mask'].cuda())
            save(OUT / 'INPUT_LOCK.json', {'front_cache_sha256': cache_sha, 'viewgen_result_sha256': sha(CS / 'views/VIEWGEN_RESULT.json'), 'labels_sha256': DIAG_LABELS_SHA})
            ev('inputs_resident', gpu_alloc_gib=torch.cuda.memory_allocated() / 2 ** 30)
            R = Recipe(torch, np); fits = {}; result['fits'] = fits
            for mode in MODES:
                for seed in SEEDS:
                    clock.check(); name = '%s_%d' % (mode, seed); d = OUT / name; d.mkdir()
                    net, priors, info = R.fit(diag, labels, gI, gV, mode, seed, clock.check, log=lambda e, **kw: ev(e, fit=name, **kw))
                    torch.save({'state_dict': net.state_dict(), 'priors': priors, 'fit': name, 'fixed_final_epoch': 12}, d / 'FINAL_EPOCH12.pt')
                    info.update(checkpoint_sha256=sha(d / 'FINAL_EPOCH12.pt'), seconds_at_end=clock.elapsed()); save(d / 'FIT.json', info)
                    fits[name] = {'checkpoint_sha256': info['checkpoint_sha256'], 'stats': info['stats'], 'final_loss': info['history'][-1]['train_loss'], 'seconds_at_end': info['seconds_at_end']}
                    ev('fit_completed', fit=name, **fits[name]); del net; torch.cuda.empty_cache()
            result.update(status='completed', fits=fits, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'E2_RESULT.json', result)


if __name__ == '__main__':
    main()
