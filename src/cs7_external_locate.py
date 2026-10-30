"""CS20261013-EXT step 1 (results/external_dataset3_20261013/PLAN.md): frozen-locator localisation of the
de-identified local dataset 3 (sagittal T2).  Per-series body is predict_sagittal.run / cs_locate.predict verbatim
except that the single sagittal T2 series of each case is passed directly (no RSNA series table).  Local only."""
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import cs_locate as L  # noqa: E402
from predict_sagittal import header, image_triplet  # noqa: E402

ROOT = Path('data/hospital/images_batch1')
OUT = Path('data/hospital/work/external_dataset3') / 'LOCALIZATION.json'


def locate(paths, model, size, root):
    heads = [header(p) for p in paths]; normal = heads[0]['n']; heads.sort(key=lambda h: (float(h['pos'] @ normal), int(h['path'].stem)))
    if any(abs(float(h['n'] @ normal)) < math.cos(math.radians(1)) for h in heads): raise ValueError('sagittal_orientation_inconsistent')
    if len({h['frame'] for h in heads}) != 1 or not heads[0]['frame']: raise ValueError('inconsistent_or_unknown_series_frame')
    unique = []
    for h in heads:
        if not unique or abs(float((h['pos'] - unique[-1]['pos']) @ normal)) > .05: unique.append(h)
    mid = len(unique) // 2; chosen = [unique[max(0, min(len(unique) - 1, mid + i))] for i in (-1, 0, 1)]; center = chosen[1]
    if len({h['path'] for h in chosen}) < 3: raise ValueError('fewer_than_3_unique_sagittal_slices')
    with torch.inference_mode(): coords = model(image_triplet([h['path'] for h in chosen], size)[None].cuda()).reshape(5, 2).cpu()
    h, w = center['shape']; dy, dx = center['sp']; aspect = w * dx / (h * dy); nh = max(1, min(size, round(size / aspect))); nw = max(1, min(size, round(size * aspect)))
    if aspect >= 1: nw = size
    else: nh = size
    left = (size - nw) // 2; top = (size - nh) // 2; sx = nw / w; sy = nh / h; q = coords * size
    original = torch.stack(((q[:, 0] - left - (sx - 1) / 2) / sx, (q[:, 1] - top - (sy - 1) / 2) / sy), dim=-1)
    in_fov = ((original[:, 0] >= 0) & (original[:, 0] < w) & (original[:, 1] >= 0) & (original[:, 1] < h)).tolist()
    return {'input_files': [h['path'].relative_to(root).as_posix() for h in chosen], 'predicted_xy_original_pixels': original.tolist(), 'predicted_in_fov': in_fov,
            'center_shape': center['shape'], 'center_pixel_spacing': center['sp'].tolist(), 'center_orientation': center['o'].tolist(),
            'center_position': center['pos'].tolist(), 'frame': center['frame'], 'unique_slices': len(unique)}


def main():
    t0 = time.time(); assert L.sha(L.LOCATOR) == L.HASHES['locator'] and not OUT.exists()
    OUT.parent.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    codes = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name.startswith('HOSP-')); assert len(codes) == 100
    ck = torch.load(L.LOCATOR, map_location='cpu', weights_only=True)
    model = resnet18(weights=None); model.fc = nn.Sequential(nn.Linear(512, 10), nn.Sigmoid()); model.load_state_dict(ck['state_dict']); model.cuda().eval(); size = ck['config']['image_size']
    rows, rejected = [], []
    for c in codes:
        try:
            r = locate(sorted((ROOT / c / 'SAG_T2').glob('*.dcm')), model, size, ROOT); r.update(study_id=c, role='external_dataset3'); rows.append(r)
        except Exception as e: rejected.append({'study_id': c, 'reason': str(e)})
    # descriptive QA: adjacent-level spacing in mm along the image plane
    gaps = []
    for r in rows:
        sp = r['center_pixel_spacing']; xy = np.array(r['predicted_xy_original_pixels']) * np.array([sp[1], sp[0]])
        gaps += np.linalg.norm(np.diff(xy, axis=0), axis=1).tolist()
    doc = {'protocol': 'CS20261013-EXT', 'predictions': rows, 'rejected': rejected, 'predicted': len(rows), 'patients': len(codes),
           'patients_all_five_in_fov': sum(all(r['predicted_in_fov']) for r in rows), 'levels_in_fov': int(sum(sum(r['predicted_in_fov']) for r in rows)),
           'adjacent_level_gap_mm': {'median': float(np.median(gaps)), 'p5': float(np.percentile(gaps, 5)), 'p95': float(np.percentile(gaps, 95)), 'min': float(min(gaps)), 'max': float(max(gaps))} if gaps else None,
           'locator_sha256': L.HASHES['locator'], 'labels_read': False, 'seconds': time.time() - t0, 'device': torch.cuda.get_device_name(0)}
    OUT.write_text(json.dumps(doc, indent=1) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in doc.items() if k != 'predictions'}))


if __name__ == '__main__':
    main()
