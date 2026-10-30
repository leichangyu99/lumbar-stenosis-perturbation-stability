"""CS20261013 R1 training: AUGMIX (CE(clean) + 12*JS3(clean, aug1, aug2)) and the compute-matched AUGCE
((CE clean + CE aug1 + CE aug2)/3), 3 seeds each, frozen grader recipe otherwise (CS20261012 section 0).
Clean views = verified front_cache resident on GPU; AugMix views streamed per effective batch from
CS/v2/augmix/eEE (SHA256 verified on every read, prefetched in a background thread)."""
import concurrent.futures as cf, hashlib, io, json, sys, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from cs_common import CACHE, CS, DIAG_LABELS, DIAG_LABELS_SHA, EPOCHS, SEEDS, Clock, GpuLock, Recipe, load, require, save, sha  # noqa: E402

OUT = HERE / 'out'
AUG = CS / 'v2/augmix'
LAMBDA = 12.0


def main():
    clock = Clock(4700); OUT.mkdir(exist_ok=False); result = {'status': 'failed', 'stage': 'CS2_R1_train'}
    log = (OUT / 'events.jsonl').open('a', encoding='utf-8')

    def ev(e, **kw):
        log.write(json.dumps({'event': e, 'elapsed_s': round(clock.elapsed(), 2), **kw}) + '\n'); log.flush()
    try:
        inv = load(HERE / 'INVENTORY.json')
        for n, h in inv['files_sha256'].items(): require(sha(HERE / n) == h, 'Inventory mismatch ' + n)
        vr = load(AUG / 'VIEWS_RESULT.json'); require(vr['status'] == 'completed' and sha(AUG / 'FILES_SHA256.json') == vr['files_index_sha256'], 'AugMix views incomplete')
        index = load(AUG / 'FILES_SHA256.json')
        diag = load(CS / 'e0/strategy_split.json')['patient_ids']['diagnostic_train']
        require(sha(DIAG_LABELS) == DIAG_LABELS_SHA, 'Labels changed'); labels = load(DIAG_LABELS)['labels']
        import numpy as np, torch
        from torch import nn
        from torch.nn import functional as F
        torch.set_num_threads(4); require(torch.cuda.is_available(), 'CUDA unavailable')
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

        def read(epoch, sids):
            out = {}
            for s in sids:
                b = (AUG / ('e%02d' % epoch) / (s + '.pt')).read_bytes(); key = 'e%02d/%s.pt' % (epoch, s)
                require(hashlib.sha256(b).hexdigest() == index[key], 'AugMix file changed ' + key)
                r = torch.load(io.BytesIO(b), map_location='cpu', weights_only=True); require(r['study_id'] == s and r['epoch'] == epoch, 'AugMix id')
                out[s] = (r['k1']['maps'], r['k1']['mask'], r['k2']['maps'], r['k2']['mask'])
            return out

        with GpuLock('CS2_train'):
            gI = {}
            for sid in diag:
                rec = torch.load(io.BytesIO((CACHE / (sid + '.pt')).read_bytes()), map_location='cpu', weights_only=True)
                require(rec['study_id'] == sid and rec['role'] == 'diagnostic_train', 'Cache ' + sid); gI[sid] = (rec['maps'].cuda(), rec['mask'].cuda())
            R = Recipe(torch, np); fits = {}; result['fits'] = fits; pool = cf.ThreadPoolExecutor(max_workers=1)
            for seed in SEEDS:
                for mode in ('AUGMIX', 'AUGCE'):
                    clock.check(); name = '%s_%d' % (mode, seed); d = OUT / name; d.mkdir()
                    ty, counts, cw, priors = R.supply(diag, labels); mean, std = R.normalisation(diag, gI)
                    torch.manual_seed(seed); np.random.seed(seed)
                    net = R.Diagnosis(R.backbone.layer4, R.SagHead().cuda(), mean, std, True).cuda()
                    h0, l0, bn0 = R.state_hash(net.head.state_dict()), R.state_hash(net.layer4.state_dict()), R.bn_state(net.layer4)
                    require(l0 == R.l4_hash, 'Non-ImageNet layer4 init')
                    opt = torch.optim.AdamW([{'params': net.layer4.parameters(), 'lr': 1e-4}, {'params': net.head.parameters(), 'lr': 1e-3}], weight_decay=.01)
                    cwc = cw.cuda(); g = torch.Generator().manual_seed(seed); hist = []; nojs = 0

                    def ce(z, tgt, known, vp):
                        per = F.cross_entropy(z.reshape(-1, 3), tgt.clamp_min(0).reshape(-1), weight=cwc, reduction='none').reshape(-1, 5) * known
                        use = known.sum(-1) > 0
                        return (per.sum(-1)[use] / known.sum(-1)[use]).sum() / vp if use.any() else None
                    for epoch in range(1, EPOCHS + 1):
                        clock.check(); net.train(); order = torch.randperm(len(diag), generator=g).tolist()
                        effs = [order[s0:s0 + 32] for s0 in range(0, len(order), 32)]; losses = []
                        fut = pool.submit(read, epoch, [diag[i] for i in effs[0]])
                        for b_i, eff in enumerate(effs):
                            data = fut.result()
                            if b_i + 1 < len(effs): fut = pool.submit(read, epoch, [diag[i] for i in effs[b_i + 1]])
                            vp = int((ty[eff] >= 0).any(-1).sum())
                            if vp == 0: continue
                            sids = [diag[i] for i in eff]
                            common = {s: (gI[s][1].any(-1).cpu() & data[s][1].any(-1) & data[s][3].any(-1)) for s in sids}
                            nc = sum(bool(c.any()) for c in common.values()); nojs += len(sids) - nc
                            opt.zero_grad(set_to_none=True); bl = 0.
                            for o in range(0, len(eff), 4):
                                ix = eff[o:o + 4]; ms = [diag[i] for i in ix]; tgt = ty[ix].cuda(); known = tgt >= 0
                                z0 = net(torch.stack([gI[s][0] for s in ms]), torch.stack([gI[s][1] for s in ms]))
                                z1 = net(torch.stack([data[s][0] for s in ms]).cuda(), torch.stack([data[s][1] for s in ms]).cuda())
                                z2 = net(torch.stack([data[s][2] for s in ms]).cuda(), torch.stack([data[s][3] for s in ms]).cuda())
                                require(bool(torch.isfinite(z0).all() and torch.isfinite(z1).all() and torch.isfinite(z2).all()), 'Nonfinite logits')
                                if mode == 'AUGCE':
                                    parts = [ce(z, tgt, known, vp) for z in (z0, z1, z2)]
                                    loss = None if parts[0] is None else sum(parts) / 3
                                else:
                                    loss = ce(z0, tgt, known, vp)
                                    both = torch.stack([common[s] for s in ms]).cuda(); cnt = both.sum(-1); ok = cnt > 0
                                    if nc > 0 and ok.any():
                                        p = [z.softmax(-1) for z in (z0, z1, z2)]; lm = ((p[0] + p[1] + p[2]) / 3).clamp(1e-7, 1).log()
                                        js3 = sum((torch.xlogy(q, q) - q * lm).sum(-1) for q in p) / 3
                                        term = LAMBDA * ((js3 * both).sum(-1)[ok] / cnt[ok]).sum() / nc
                                        loss = term if loss is None else loss + term
                                if loss is not None: loss.backward(); bl += float(loss.detach())
                            nn.utils.clip_grad_norm_(net.layer4.parameters(), 5, error_if_nonfinite=True)
                            nn.utils.clip_grad_norm_(net.head.parameters(), 5, error_if_nonfinite=True)
                            opt.step(); losses.append(bl)
                        require(losses and all(bool(torch.isfinite(x).all()) for x in net.parameters()), 'Nonfinite weights')
                        require(R.same_state(bn0, R.bn_state(net.layer4)), 'BN statistics changed')
                        hist.append({'epoch': epoch, 'train_loss': sum(losses) / len(losses)}); ev('epoch', fit=name, **hist[-1])
                    require(R.state_hash(net.layer4.state_dict()) != l0 and R.state_hash(net.head.state_dict()) != h0, 'No update')
                    torch.save({'state_dict': net.state_dict(), 'priors': priors, 'fit': name, 'fixed_final_epoch': 12}, d / 'FINAL_EPOCH12.pt')
                    info = {'mode': mode, 'seed': seed, 'lambda': LAMBDA if mode == 'AUGMIX' else None, 'history': hist, 'patients_without_common_level_batches': nojs,
                            'class_counts': counts.tolist(), 'checkpoint_sha256': sha(d / 'FINAL_EPOCH12.pt'), 'seconds_at_end': clock.elapsed()}
                    save(d / 'FIT.json', info); fits[name] = {k: info[k] for k in ('checkpoint_sha256', 'seconds_at_end', 'patients_without_common_level_batches')}
                    fits[name]['final_loss'] = hist[-1]['train_loss']; ev('fit_completed', fit=name, **fits[name]); del net, opt; torch.cuda.empty_cache()
            result.update(status='completed', peak_gpu_gib=torch.cuda.max_memory_allocated() / 2 ** 30)
    except BaseException as exc:
        result.update(reason=str(exc), traceback=traceback.format_exc()); raise
    finally:
        result['seconds'] = clock.elapsed(); save(OUT / 'TRAIN_RESULT.json', result)


if __name__ == '__main__':
    main()
