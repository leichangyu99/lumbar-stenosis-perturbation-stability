"""CS20261013 AugMix training views: 1107 diagnostic_train x 12 epochs x 2 views, shared across seeds.
Output CS/v2/augmix/eEE/<sid>.pt {'k1':{maps,mask},'k2':{...},'fallback':n}; per-file SHA256 from the written bytes."""
import hashlib, io, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, DATA, PRED, WEIGHTS, Clock, GpuLock, load, require, save, sha  # noqa: E402

OUT = CS / 'v2/augmix'


def main():
    clock = Clock(4700); OUT.mkdir(parents=True, exist_ok=False); result = {'status': 'failed', 'stage': 'CS2_augmix_views', 'labels_read': False}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        diag = load(CS / 'e0/strategy_split.json')['patient_ids']['diagnostic_train']
        rows = {r['study_id']: r for r in load(PRED)['predictions'] if r['role'] == 'diagnostic_train'}
        import torch
        import cs_engine as E, cs2_engine as E2
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        for e in range(1, 13): (OUT / ('e%02d' % e)).mkdir()
        files = {}; fallback = 0; missing = []
        with GpuLock('CS2_views'):
            _, prefix = E.load_prefix(WEIGHTS, 'cuda')
            for i, sid in enumerate(diag):
                clock.check()
                if sid in rows:
                    P = E2.Patient(DATA, rows[sid], sid); clean = P.crops(E.IDENTITY)
                else:
                    P = None; missing.append(sid)
                for e in range(1, 13):
                    rec = {'study_id': sid, 'epoch': e}
                    for k in (1, 2):
                        if P is None:
                            maps, mask, fb = torch.zeros(5, 3, 256, 14, 14), torch.zeros(5, 3, dtype=torch.bool), 0
                        else:
                            maps, mask, fb = P.augmix(clean, E2.augmix_spec(sid, e, k), prefix)
                        require(bool(torch.isfinite(maps).all()), 'Nonfinite augmix ' + sid); fallback += fb
                        rec['k%d' % k] = {'maps': maps, 'mask': mask}
                    buf = io.BytesIO(); torch.save(rec, buf); b = buf.getvalue()
                    p = OUT / ('e%02d' % e) / (sid + '.pt'); p.write_bytes(b); files['e%02d/%s.pt' % (e, sid)] = hashlib.sha256(b).hexdigest()
                if (i + 1) % 100 == 0: ev('patients', n=i + 1)
        save(OUT / 'FILES_SHA256.json', files)
        result.update(status='completed', patients=len(diag), files=len(files), chain_fallbacks=fallback, missing_sagittal=missing, files_index_sha256=sha(OUT / 'FILES_SHA256.json'))
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'VIEWS_RESULT.json', result)


if __name__ == '__main__':
    main()
