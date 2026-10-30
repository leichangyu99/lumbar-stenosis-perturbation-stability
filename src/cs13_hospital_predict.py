"""HOSPITAL-257 (results/external_257_v1cross_20261021/PLAN.md): frozen locator + 21 models on HOSP-101..257
(de-identified images, read-only).  Reuses cs7_external_locate.locate and cs7_external_predict.main unchanged,
only pointing ROOT/WORK to the batch-2 locations.  Outputs are written to local storage only."""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cs_locate as L  # noqa: E402
import cs7_external_locate as XL  # noqa: E402
import cs7_external_predict as XP  # noqa: E402

ROOT = Path('data/hospital/images_batch2')
WORK = Path('data/hospital/work/external_dataset3_batch2')


def locate_all():
    out = WORK / 'LOCALIZATION.json'; assert not out.exists()
    WORK.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4); t0 = time.time()
    codes = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith('HOSP-')); assert len(codes) == 157 and codes[0] == 'HOSP-101' and codes[-1] == 'HOSP-257'
    ck = torch.load(L.LOCATOR, map_location='cpu', weights_only=True); assert L.sha(L.LOCATOR) == L.HASHES['locator']
    model = resnet18(weights=None); model.fc = nn.Sequential(nn.Linear(512, 10), nn.Sigmoid()); model.load_state_dict(ck['state_dict']); model.cuda().eval(); size = ck['config']['image_size']
    rows, rejected = [], []
    for c in codes:
        try:
            r = XL.locate(sorted((ROOT / c / 'SAG_T2').glob('*.dcm')), model, size, ROOT); r.update(study_id=c, role='external_dataset3_batch2'); rows.append(r)
        except Exception as e: rejected.append({'study_id': c, 'reason': str(e)})
    gaps = []
    for r in rows:
        sp = r['center_pixel_spacing']; xy = np.array(r['predicted_xy_original_pixels']) * np.array([sp[1], sp[0]]); gaps += np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist()
    doc = {'protocol': 'HOSPITAL-257', 'predictions': rows, 'rejected': rejected, 'predicted': len(rows), 'patients': len(codes),
           'patients_all_five_in_fov': sum(all(r['predicted_in_fov']) for r in rows), 'levels_in_fov': int(sum(sum(r['predicted_in_fov']) for r in rows)),
           'adjacent_level_gap_mm': {'median': float(np.median(gaps)), 'p5': float(np.percentile(gaps, 5)), 'p95': float(np.percentile(gaps, 95))} if gaps else None,
           'locator_sha256': L.HASHES['locator'], 'labels_read': False, 'seconds': time.time() - t0}
    out.write_text(json.dumps(doc, indent=1) + '\n', encoding='utf-8'); print(json.dumps({k: v for k, v in doc.items() if k != 'predictions'}))


if __name__ == '__main__':
    locate_all()
    XP.ROOT = ROOT; XP.WORK = WORK
    XP.main()
