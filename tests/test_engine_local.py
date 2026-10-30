"""Local engine tests: intervention conventions on synthetic slices, and bitwise identity against the
original crop_image + front_cache loop on real development DICOMs (CPU prefix)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
AUTOMATION = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(AUTOMATION))
import cs_engine as E  # noqa: E402

DATA = Path('data/LumbarDISC/dataset')
PREDICTIONS = AUTOMATION.parent / 'runs/local/M1_sag_export_s20261007/attempt_0/predictions.json'
WEIGHTS = AUTOMATION / 'weights/resnet18-f37072fd.pth'


class FakeDS:
    def __init__(self, image, dy=.5, dx=.5):
        self.pixel_array = image; self.Rows, self.Columns = image.shape
        self.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; self.ImagePositionPatient = [0, 0, 0]
        self.PixelSpacing = [dy, dx]; self.PhotometricInterpretation = 'MONOCHROME2'


def view_crop(ds, x, y, v, sid='S', j=0):
    point = np.array([x * float(ds.PixelSpacing[1]), y * float(ds.PixelSpacing[0]), 0.])
    c = E.centre(ds, point, v)
    return None if c is None else E.crop(E.intensity(ds, v['window'], v['sigma'], E.noise_seed(sid, j)), *c, v)[0]


def synthetic_tests():
    cols = np.tile(np.arange(512, dtype=np.uint16), (512, 1)); ds = FakeDS(cols)
    lo, hi = np.percentile(cols.astype(np.float32), [1, 99]); step = 1 / (hi - lo) / .229
    base = view_crop(ds, 256, 256, E.IDENTITY)
    inner = (slice(20, -20), slice(20, -20))
    tc = view_crop(ds, 256, 256, E.view('t', 'T', tx=1.0)); tr = view_crop(ds, 256, 256, E.view('t', 'T', ty=1.0))
    assert torch.allclose((tc - base)[inner], torch.full_like(base[inner], float(step)), atol=1e-4), 'T column shift'
    assert torch.allclose(tr[inner], base[inner], atol=1e-5), 'T row shift must not change a column ramp'
    r90 = view_crop(ds, 256, 256, E.view('r', 'R', theta_deg=90.0))
    assert float((r90[inner][:, 1:] - r90[inner][:, :-1]).abs().max()) < 1e-3, 'R90: constant along output columns'
    assert bool(((r90[inner][1:, :] - r90[inner][:-1, :]) < 0).all()), 'R+90: column axis rotated onto rows (values fall down rows)'
    r_small = view_crop(ds, 256, 256, E.view('r', 'R', theta_deg=4.0))
    assert not torch.equal(r_small, base) and float((r_small - base)[inner].abs().max()) < 1.0
    s_half = view_crop(ds, 256, 256, E.view('s', 'S', s=0.5))
    slope = lambda t: float((t[inner][:, 1:] - t[inner][:, :-1]).mean())
    assert abs(slope(s_half) / slope(base) - 0.5) < 1e-3, 'S scales the field of view'
    w = view_crop(ds, 256, 256, E.view('w', 'W', window=(5, 95)))
    lo5, hi5 = np.percentile(cols.astype(np.float32), [5, 95])
    x = (base * .229 + .485) * float(hi - lo) + float(lo)
    expect = (((x - float(lo5)) / float(hi5 - lo5)) - .485) / .229
    assert torch.allclose(w[inner], expect[inner], atol=1e-3), 'W window'
    img0 = E.intensity(ds, (1, 99), 0.0, 0); a = E.intensity(ds, (1, 99), .05, E.noise_seed('S', 0)); b = E.intensity(ds, (1, 99), .10, E.noise_seed('S', 0))
    interior = (img0 > .4) & (img0 < .6) & (a > 0) & (a < 1) & (b > 0) & (b < 1)
    assert np.allclose(((a - img0) / .05)[interior], ((b - img0) / .10)[interior], atol=1e-4), 'N dose coupling (same Z)'
    assert not np.array_equal(E.intensity(ds, (1, 99), .05, E.noise_seed('S', 1)), a), 'N seed per slice'
    assert np.array_equal(E.intensity(ds, (1, 99), .05, E.noise_seed('S', 0)), a), 'N reproducible'
    assert view_crop(ds, 511.6, 256, E.view('t', 'T', tx=1.0)) is None, 'centre outside the slice -> invisible'
    m = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 0], [1, 1, 0]], dtype=torch.bool)
    dc, kc = E.drop_slice(m, 'center'); de, ke = E.drop_slice(m, 'edge')
    assert dc.tolist() == [[1, 0, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0]] and kc == 1
    assert de.tolist() == [[0, 1, 1], [0, 0, 1], [0, 1, 0], [0, 0, 0], [0, 1, 0]] and ke == 1
    v1 = E.train_view('12345'); v2 = E.train_view('12345'); v3 = E.train_view('12346')
    assert v1 == v2 and v1 != v3 and -1 <= v1['tx'] <= 1 and -1 <= v1['ty'] <= 1 and -4 <= v1['theta_deg'] <= 4 and .94 <= v1['s'] <= 1.06
    views = E.evaluation_views(); assert [v['family'] for v in views].count('T') == 6 and len(views) == 27
    print('synthetic engine tests passed')


def reference_cache(row, prefix):
    """Verbatim front_cache loop from train_sag_layer4_pair.run (device-parameterised)."""
    from extract_sagittal_rois import crop_image
    import pydicom
    maps = torch.zeros(5, 3, 256, 14, 14); mask = torch.zeros(5, 3, dtype=torch.bool)
    dslist = [pydicom.dcmread(DATA / f) for f in row['input_files']]; o = np.array(row['center_orientation']); sp = row['center_pixel_spacing']; pos = np.array(row['center_position']); crops = []; locations = []
    for k, (xy, valid) in enumerate(zip(row['predicted_xy_original_pixels'], row['predicted_in_fov'])):
        if not valid: continue
        point = pos + o[:3] * xy[0] * sp[1] + o[3:] * xy[1] * sp[0]
        for j, ds in enumerate(dslist):
            image = crop_image(ds, point, 224, 64)
            if image is not None: crops.append(image); locations.append((k, j))
    if crops:
        with torch.inference_mode(): encoded = torch.cat([prefix(torch.stack(crops[j:j + 16]).cpu()).cpu() for j in range(0, len(crops), 16)])
        for location, feature in zip(locations, encoded): maps[location] = feature; mask[location] = True
    return maps, mask


def real_identity_tests(n=2):
    torch.set_num_threads(4)
    rows = {r['study_id']: r for r in json.loads(PREDICTIONS.read_text(encoding='utf-8'))['predictions'] if r['role'] == 'diagnostic_train'}
    _, prefix = E.load_prefix(WEIGHTS, 'cpu')
    for sid in sorted(rows)[:n]:
        ref_maps, ref_mask = reference_cache(rows[sid], prefix)
        out = E.build_views(DATA, rows[sid], sid, [E.evaluation_views()[0]], prefix, 'cpu')
        maps, mask, _ = out['I']
        assert torch.equal(maps, ref_maps) and torch.equal(mask, ref_mask), 'identity view differs from front_cache loop ' + sid
        print('identity bitwise equal', sid, int(mask.sum()), 'slices')


if __name__ == '__main__':
    synthetic_tests()
    real_identity_tests()
