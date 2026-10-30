"""Descriptive per-model panels for the manuscript (post hoc).
P1 accuracy-stability plane: for each of the 18 single models (A0-A5 x 3 seeds), each 3-seed configuration ensemble and the
   full 18-model ensemble: 1-px flip (direction mean), patient RPS, level accuracy and severe recall on identity view.
P2 grade-pair composition of 1-px flips (normal<->moderate, moderate<->severe, normal<->severe), single-model mean and ensemble.
P3 per-view flip for every configuration ensemble on the four cohorts (dose curves for all graders).
Parameter counts of the backbones are computed from the architecture definitions (no weights needed)."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs15_ensemble as E  # noqa: E402
from cs_common import rps_patient  # noqa: E402
from cs_metrics import grade  # noqa: E402

OUT = HERE.parent.parent / 'results/clinical_addons_20261021/PAPER_PANELS.json'
T1 = ('T_1_col', 'T_1_row')


def acc_metrics(P, vn, Y):
    I = vn.index('I'); p = P[:, I]; g = grade(p); k = Y >= 0
    rps = [rps_patient(Y[i], p[i]) for i in range(len(Y))]; rps = [r for r in rps if r is not None]
    return {'RPS': float(np.mean(rps)), 'level_accuracy': float((g[k] == Y[k]).mean()), 'severe_recall': float((g[Y == 2] == 2).mean())}


def pair_counts(P, vn, v0):
    I = vn.index('I'); g0 = grade(P[:, I]); out = {'normal_moderate': 0, 'moderate_severe': 0, 'normal_severe': 0}
    for v in T1:
        g1 = grade(P[:, vn.index(v)]); f = (g0 != g1) & v0.astype(bool)
        for a, b in zip(g0[f], g1[f]):
            key = {frozenset((0, 1)): 'normal_moderate', frozenset((1, 2)): 'moderate_severe', frozenset((0, 2)): 'normal_severe'}[frozenset((int(a), int(b)))]
            out[key] += 0.5
    return out


def per_view(P, vn, v0):
    I = vn.index('I'); g0 = grade(P[:, I]); c = np.maximum(v0.sum(-1), 1)
    return {v: float((((g0 != grade(P[:, vn.index(v)])) * v0).sum(-1) / c).mean()) for v in vn if v != 'I'}


def params():
    import torchvision.models as tv
    out = {}
    for k, f in (('ResNet18', tv.resnet18), ('ConvNeXt-Tiny', tv.convnext_tiny), ('EfficientNetV2-S', tv.efficientnet_v2_s), ('Swin-T', tv.swin_t), ('ResNet50', tv.resnet50)):
        m = f(weights=None); out[k] = int(sum(p.numel() for p in m.parameters()))
    return out


def main():
    res = {'note': 'post hoc descriptive panels', 'backbone_parameters_incl_imagenet_head': params()}
    for which in ('validation', 'internal671', 'hospital256', 'spider'):
        P, v0, vn, Y = E.load(which); comp = v0.any(-1); P = P[:, comp]; v0 = v0[comp]; Y = Y[comp] if Y is not None else None
        r = {'single': {}, 'arch3': {}, 'full': {}}
        for k, name in enumerate(E.NAMES):
            r['single'][name] = {'flip': float(E.flip(P[k], vn, v0, T1).mean())}
            if Y is not None:
                r['single'][name].update(acc_metrics(P[k], vn, Y))
        for i, a in enumerate(E.ARCHS):
            Pa = P[3 * i:3 * i + 3].mean(0); r['arch3'][a] = {'flip': float(E.flip(Pa, vn, v0, T1).mean()), 'per_view': per_view(Pa, vn, v0)}
            if Y is not None:
                r['arch3'][a].update(acc_metrics(Pa, vn, Y))
        Pf = P.mean(0); r['full'] = {'flip': float(E.flip(Pf, vn, v0, T1).mean()), 'per_view': per_view(Pf, vn, v0), 'pairs': pair_counts(Pf, vn, v0)}
        if Y is not None:
            r['full'].update(acc_metrics(Pf, vn, Y))
        pc = [pair_counts(P[k], vn, v0) for k in range(18)]; r['single_pairs_mean'] = {k: float(np.mean([x[k] for x in pc])) for k in pc[0]}
        r['levels_evaluable'] = int(v0.sum()); res[which] = r
        print(which, 'full', {k: v for k, v in r['full'].items() if k not in ('per_view',)}, 'pairs single', r['single_pairs_mean'])
    OUT.write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8'); print('params', res['backbone_parameters_incl_imagenet_head'])


if __name__ == '__main__':
    main()
