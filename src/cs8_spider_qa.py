"""CS20261013-SPX descriptive QA: localisation versus SPIDER expert disc masks (lowest five IVD centroids ordered along the
superior axis) and the thin-slice (<=2 mm) subgroup T 1px flip per system.  Masks are used for QA only."""
import json, re, sys, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs8_spider_convert as CV  # noqa: E402
import cs8_spider_stats as ST  # noqa: E402

MASKS = Path('data/public/SPIDER_masks_20261012/masks.zip')
WORK = ST.WORK
OUT = ST.OUT


def main():
    loc = json.loads((WORK / 'LOCALIZATION.json').read_text(encoding='utf-8')); rows = {r['study_id']: r for r in loc['predictions']}
    conv = json.loads((CV.OUT / 'CONVERT_REPORT.json').read_text(encoding='utf-8'))
    z = zipfile.ZipFile(MASKS); dist = {k: [] for k in range(5)}; order_ok = []; cases = 0; few = 0
    for case, r in rows.items():
        name = 'masks/%d_t2.mha' % int(case[3:]); arr, O, sp, D, _ = CV.read_mha(z.read(name)); vol = np.transpose(arr, (2, 1, 0))
        labs = [int(l) for l in np.unique(vol) if 201 <= l <= 250]
        cents = []
        for l in labs:
            idx = np.argwhere(vol == l); cents.append(O + (idx.mean(0) * sp) @ D)
        if len(cents) < 5: few += 1; continue
        cents = sorted(cents, key=lambda c: c[2])[:5][::-1]            # lowest five, ordered cranial->caudal = L1/L2..L5/S1
        o = np.array(r['center_orientation']); s = r['center_pixel_spacing']; pos = np.array(r['center_position'])
        pts = [pos + o[:3] * xy[0] * s[1] + o[3:] * xy[1] * s[0] for xy in r['predicted_xy_original_pixels']]
        cases += 1; ok = True
        for k, p in enumerate(pts):
            d2 = [np.linalg.norm((p - c)[1:]) for c in cents]; dist[k].append(d2[k]); ok &= int(np.argmin(d2)) == k
        order_ok.append(ok)
    qa = {'cases_with_5_discs': cases, 'cases_fewer_than_5_discs': few,
          'distance_to_expected_disc_mm': {('L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1')[k]: {'median': float(np.median(v)), 'p90': float(np.percentile(v, 90)), 'frac_le_10mm': float(np.mean(np.array(v) <= 10))} for k, v in dist.items()},
          'all_five_nearest_to_expected_disc': float(np.mean(order_ok))}
    # thin-slice subgroup
    d = np.load(WORK / 'PRED_spider.npz'); names = [str(n) for n in d['models']]; keys = [str(k) for k in d['keys']]; codes = [str(c) for c in d['patients']]
    vn = [k.split('|')[0] for k in keys if k.endswith('|None')]; nv = len(vn); thin = np.array([conv[c].get('thickness_mm', 99) <= 2 for c in codes])
    sub = {}
    for s in ST.SYSTEMS:
        P = d['probs'][[k for k, n in enumerate(names) if n.startswith(s + '_')]][:, :, :nv].mean(0)
        for tag, m in (('thin_le_2mm', thin), ('thick_gt_2mm', ~thin)):
            pv, fam, comp = ST.family_flips(P[m], vn, d['visible'][m][:, vn.index('I')])
            sub.setdefault(s, {})[tag] = {'patients': int(m.sum()), 'T1px_flip': float(((pv['T_1_col'][0] + pv['T_1_row'][0]) / 2).mean())}
    res = {'localization_QA': qa, 'thickness_subgroups': sub, 'conversion_failures': {k: v['error'] for k, v in conv.items() if 'error' in v}}
    (OUT / 'SPX_QA.json').write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8'); print(json.dumps(res, indent=1)[:3000])


if __name__ == '__main__':
    main()
