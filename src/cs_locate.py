"""CS20261012 E0: localise the full 197-patient source_confirmation role with the frozen locator.

The per-patient body is predict_sagittal.run verbatim (series choice, median physical plane triplet,
locator inference, back-projection, FOV flags); only the role selection differs.  No coordinate or
grade CSV is read.  A reproduction check reruns 20 development patients and compares with the
original sagittal_predictions.json rows (the same local machine produced them).
"""
import csv, hashlib, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18

HERE = Path(__file__).resolve().parent
AUTOMATION = HERE.parent
sys.path.insert(0, str(AUTOMATION))
from predict_sagittal import header, image_triplet  # noqa: E402

DATA = Path('data/LumbarDISC/dataset')
SPLIT = AUTOMATION / 'configs/strategy_split.json'
LOCATOR = AUTOMATION.parent / 'runs/remote_mirror/M1_locator_xy_s20261007/best_locator.pt'
ORIGINAL = AUTOMATION.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
SUBSET300 = AUTOMATION / 'sr20261010/recovery_r4/r5_stable_20261012/final_mirror_20261012/cohort_control/ALL300_TARGETS.json'
OUT = HERE / 'e0/CONFIRMATION_LOCALIZATION.json'
HASHES = {'split': 'ebd8382a3a0cd08f32c9cea56b81dd97d9cc1b8ccde280b91b8a2f7b7ee1a7b1', 'locator': 'fba2ec646032d0b4dd6fb585332abe3390110fe3be3a825c35a7e723401182a8',
          'original': '2a18dcfcfbe53249123ed44a4d95f97a9286d0268e52b3c06c5a5e3dbf8b659c', 'predict_sagittal': '0f2d7bb6f47e183340d48fe7dda9358cd7b37834cd1cc8cac0dd928e0618f803',
          'subset300': '196467fea869b04ea074e54ea77406cc95e4b5a61ac2732bb6c2e44c9e9de17c'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dev20(ids):
    return sorted(ids, key=lambda s: hashlib.sha256(('CS20261012|e0_dev|' + s).encode()).hexdigest())[:20]


def predict(selected, model, size, series, root):
    rows = []; reject = []
    for study, role in selected.items():
        try:
            choices = [(len(paths := list((root / 'train_images' / study / sid).glob('*.dcm'))), -int(sid), sid, paths) for sid in series[study]]
            if not choices: raise ValueError('missing_sagittal')
            _, _, sid, paths = max(choices, key=lambda x: (x[0], x[1])); heads = [header(p) for p in paths]; normal = heads[0]['n']; heads.sort(key=lambda h: (float(h['pos'] @ normal), int(h['path'].stem)))
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
            rows.append({'study_id': study, 'role': role, 'series_id': sid, 'center_file': center['path'].relative_to(root).as_posix(), 'input_files': [h['path'].relative_to(root).as_posix() for h in chosen],
                         'predicted_xy_original_pixels': original.tolist(), 'predicted_in_fov': in_fov, 'center_shape': center['shape'], 'center_pixel_spacing': center['sp'].tolist(),
                         'center_orientation': center['o'].tolist(), 'center_position': center['pos'].tolist(), 'frame': center['frame']})
        except Exception as e: reject.append({'study_id': study, 'role': role, 'reason': str(e)})
    return rows, reject


def main():
    t0 = time.time()
    for k, p in [('split', SPLIT), ('locator', LOCATOR), ('original', ORIGINAL), ('predict_sagittal', AUTOMATION / 'predict_sagittal.py'), ('subset300', SUBSET300)]:
        assert sha(p) == HASHES[k], k
    assert not OUT.exists(), 'confirmation localisation already exists'
    torch.set_num_threads(4)
    roles = json.loads(SPLIT.read_text(encoding='utf-8-sig'))['patient_ids']
    sets = [set(x) for x in roles.values()]; assert all(not (x & y) for i, x in enumerate(sets) for y in sets[i + 1:])
    conf = roles['source_confirmation']; assert len(conf) == len(set(conf)) == 197
    dev = dev20(list(json.loads(SUBSET300.read_text(encoding='utf-8-sig'))))
    assert set(dev) <= set(roles['diagnostic_train'])
    selected = {s: 'source_confirmation' for s in conf}; selected.update({s: 'diagnostic_train' for s in dev})
    series = defaultdict(list)
    for row in csv.DictReader((DATA / 'train_series_descriptions.csv').open(encoding='utf-8-sig')):
        if row['study_id'] in selected and row['series_description'] == 'Sagittal T2/STIR': series[row['study_id']].append(row['series_id'])
    checkpoint = torch.load(LOCATOR, map_location='cpu', weights_only=True)
    assert checkpoint['config']['split_sha256'] == HASHES['split'], 'locator split mismatch'
    model = resnet18(weights=None); model.fc = nn.Sequential(nn.Linear(512, 10), nn.Sigmoid()); model.load_state_dict(checkpoint['state_dict']); model.cuda().eval(); size = checkpoint['config']['image_size']
    t1 = time.time()
    dev_rows, dev_reject = predict({s: 'diagnostic_train' for s in dev}, model, size, series, DATA)
    t2 = time.time()
    conf_rows, conf_reject = predict({s: 'source_confirmation' for s in conf}, model, size, series, DATA)
    t3 = time.time()
    original = {r['study_id']: r for r in json.loads(ORIGINAL.read_text(encoding='utf-8'))['predictions']}
    repro = []
    for r in dev_rows:
        o = original[r['study_id']]
        diff = float(np.abs(np.array(r['predicted_xy_original_pixels']) - np.array(o['predicted_xy_original_pixels'])).max())
        same_other = all(r[k] == o[k] for k in ('series_id', 'input_files', 'predicted_in_fov', 'center_shape', 'center_pixel_spacing', 'center_orientation', 'center_position', 'frame'))
        repro.append({'study_id': r['study_id'], 'max_abs_xy_diff_px': diff, 'exact_xy': r['predicted_xy_original_pixels'] == o['predicted_xy_original_pixels'], 'same_geometry_and_flags': same_other})
    finite = all(np.isfinite(r['predicted_xy_original_pixels']).all() for r in conf_rows)
    doc = {'protocol': 'CS20261012', 'stage': 'E0_confirmation_localization', 'role': 'source_confirmation', 'patients_in_role': 197,
           'predictions': conf_rows, 'rejected': conf_reject, 'predicted': len(conf_rows),
           'levels_in_fov': int(sum(sum(r['predicted_in_fov']) for r in conf_rows)), 'patients_all_five_in_fov': sum(all(r['predicted_in_fov']) for r in conf_rows),
           'finite': bool(finite), 'coordinate_csv_loaded': False, 'grade_csv_loaded': False, 'labels_read': False,
           'split_sha256': HASHES['split'], 'locator_sha256': HASHES['locator'], 'source_sha256': sha(__file__), 'predict_sagittal_sha256': HASHES['predict_sagittal'],
           'device': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'reproduction_dev20': {'patients': dev, 'rejected': dev_reject, 'rows': repro,
                                  'all_exact_xy': all(x['exact_xy'] for x in repro) and len(repro) == 20, 'all_same_geometry': all(x['same_geometry_and_flags'] for x in repro),
                                  'max_abs_xy_diff_px': max((x['max_abs_xy_diff_px'] for x in repro), default=None)},
           'executable': len(conf_rows) > 0, 'seconds': {'setup': t1 - t0, 'dev20': t2 - t1, 'confirmation197': t3 - t2, 'total': time.time() - t0}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.tmp'); tmp.write_text(json.dumps(doc, indent=1) + '\n', encoding='utf-8'); tmp.replace(OUT)
    print(json.dumps({k: v for k, v in doc.items() if k not in ('predictions',)}, default=str)[:3000])


if __name__ == '__main__':
    main()
