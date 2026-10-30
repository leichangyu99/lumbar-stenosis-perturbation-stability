"""CS20261013-MA training: winner-style graders (cs3_arch.ARCHS) x seeds on diagnostic_train 1107, full
fine-tuning with GPU ShiftScaleRotate + brightness/contrast, bf16 autocast, frozen recipe otherwise
(patient-equal weighted CE, sqrt inverse-frequency class weights, AdamW 1e-4/1e-3 wd .01, clip 5,
effective batch 32 / microbatch 4, fixed 12 epochs).  Development RPS on model_selection 197 only."""
import argparse, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CS, DATA, DIAG_LABELS, DIAG_LABELS_SHA, EPOCHS, PRED, SEEDS, Clock, GpuLock, labels_from_train_csv, load, require, rps_patient, save, sha  # noqa: E402

WEIGHTS = CS / 'ma/weights'


def build_crops(ids, role, torch, ev):
    import cs2_engine as E2, cs3_arch as A, cs_engine as E
    rows = {r['study_id']: r for r in load(PRED)['predictions'] if r['role'] == role}
    raw = torch.zeros(len(ids), 5, 3, 224, 224, dtype=torch.float16); mask = torch.zeros(len(ids), 5, 3, dtype=torch.bool); missing = []
    for i, sid in enumerate(ids):
        if sid not in rows: missing.append(sid); continue
        r, m = A.patient_raw(E2.Patient(DATA, rows[sid], sid), E.IDENTITY); raw[i] = r.half(); mask[i] = m
    ev('crops_built', role=role, n=len(ids), missing=len(missing))
    return raw.cuda(), mask.cuda(), missing


def main():
    a = argparse.ArgumentParser(); a.add_argument('--archs', nargs='+', required=True); a.add_argument('--tag', required=True); args = a.parse_args()
    out = HERE / ('out_' + args.tag); clock = Clock(9000); out.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'MA_train_' + args.tag, 'archs': args.archs}
    log = (out / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        import numpy as np, torch
        from torch import nn
        from torch.nn import functional as F
        import cs3_arch as A
        for arch in args.archs: fn, h = A.WEIGHT_FILES[A.WEIGHT_KEY[arch]]; require(sha(WEIGHTS / fn) == h, 'Weights differ ' + fn)
        split = load(CS / 'e0/strategy_split.json')['patient_ids']; diag = split['diagnostic_train']; sel = split['model_selection']
        require(sha(DIAG_LABELS) == DIAG_LABELS_SHA, 'Labels changed'); labels = load(DIAG_LABELS)['labels']
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        with GpuLock('MA_train_' + args.tag):
            Xd, Md, miss = build_crops(diag, 'diagnostic_train', torch, ev); Xs, Ms, _ = build_crops(sel, 'model_selection', torch, ev)
            sl = labels_from_train_csv(sel); Ys = np.array([sl[s] for s in sel])
            ty = torch.tensor([labels[s] for s in diag], dtype=torch.long)
            counts = torch.bincount(ty[ty >= 0], minlength=3); cw = (counts.sum() / (3 * counts.float().clamp_min(1))).sqrt(); cw /= cw.mean()
            lc = torch.stack([torch.bincount(ty[:, j][ty[:, j] >= 0], minlength=3) for j in range(5)]); priors = (lc.float() + 1) / (lc.sum(-1).float()[:, None] + 3)
            fits = {}; result['fits'] = fits
            for arch in args.archs:
                enc = A.ARCHS[arch][0]
                for seed in SEEDS:
                    clock.check(); name = '%s_%d' % (arch, seed); d = out / name; d.mkdir()
                    torch.manual_seed(seed); np.random.seed(seed)
                    net = A.Grader(arch, WEIGHTS / A.weight_file(arch)).cuda()
                    opt = torch.optim.AdamW([{'params': net.enc.parameters(), 'lr': 1e-4}, {'params': net.head_params(), 'lr': 1e-3}], weight_decay=.01)
                    g = torch.Generator().manual_seed(seed); ga = torch.Generator(device='cuda').manual_seed(seed); cwc = cw.cuda(); hist = []
                    for epoch in range(1, EPOCHS + 1):
                        clock.check(); net.train(); order = torch.randperm(len(diag), generator=g).tolist(); losses = []
                        for s0 in range(0, len(order), 32):
                            eff = order[s0:s0 + 32]; vp = int((ty[eff] >= 0).any(-1).sum())
                            if vp == 0: continue
                            opt.zero_grad(set_to_none=True); bl = 0.
                            for o in range(0, len(eff), 4):
                                ix = eff[o:o + 4]; tgt = ty[ix].cuda(); known = tgt >= 0; it = torch.tensor(ix, device='cuda')
                                x = A.augment(Xd[it], ga)
                                with torch.autocast('cuda', dtype=torch.bfloat16): z = net(x, Md[it])
                                z = z.float(); require(bool(torch.isfinite(z).all()), 'Nonfinite logits ' + name)
                                per = F.cross_entropy(z.reshape(-1, 3), tgt.clamp_min(0).reshape(-1), weight=cwc, reduction='none').reshape(-1, 5) * known
                                use = known.sum(-1) > 0
                                if use.any():
                                    loss = (per.sum(-1)[use] / known.sum(-1)[use]).sum() / vp; loss.backward(); bl += float(loss.detach())
                            nn.utils.clip_grad_norm_(net.enc.parameters(), 5, error_if_nonfinite=True)
                            nn.utils.clip_grad_norm_(net.head_params(), 5, error_if_nonfinite=True)
                            opt.step(); losses.append(bl)
                        hist.append({'epoch': epoch, 'train_loss': sum(losses) / len(losses)}); ev('epoch', fit=name, **hist[-1])
                    net.eval(); pr = []
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        for s0 in range(0, len(sel), 8):
                            z = net(Xs[s0:s0 + 8].float(), Ms[s0:s0 + 8]).float().softmax(-1).cpu().double()
                            vis = Ms[s0:s0 + 8].any(-1).cpu(); pr.append(torch.where(vis[..., None], z, priors.double()[None].expand(len(z), -1, -1)))
                    P = torch.cat(pr).numpy(); rp = [rps_patient(Ys[i], P[i]) for i in range(len(sel))]; dev_rps = float(np.mean([x for x in rp if x is not None]))
                    torch.save({'state_dict': net.state_dict(), 'priors': priors, 'arch': arch, 'seed': seed, 'fixed_final_epoch': EPOCHS}, d / 'FINAL_EPOCH12.pt')
                    fits[name] = {'arch': arch, 'encoder': enc, 'weights': A.weight_file(arch), 'seed': seed, 'dev_RPS_model_selection': dev_rps, 'final_loss': hist[-1]['train_loss'],
                                  'checkpoint_sha256': sha(d / 'FINAL_EPOCH12.pt'), 'seconds_at_end': clock.elapsed()}
                    save(d / 'FIT.json', dict(fits[name], history=hist)); ev('fit_completed', fit=name, **fits[name]); del net, opt; torch.cuda.empty_cache()
            result.update(status='completed', diag_missing_sagittal=miss, peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(out / 'TRAIN_RESULT.json', result)


if __name__ == '__main__':
    main()
