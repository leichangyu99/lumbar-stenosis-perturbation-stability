"""CS20261012 frozen computational input-intervention engine (EXPERIMENT_PLAN V3 section 2 and 10).

The identity view reproduces extract_sagittal_rois.crop_image and the train_sag_layer4_pair
front_cache construction operation by operation: same decode, percentile window, grid, bilinear
zero-padded grid_sample, ImageNet normalisation, crop order (level k, then slice j) and CUDA prefix
batches of 16.  Interventions only change parameters that are neutral (0, 1, [1,99]) for identity.
"""
import hashlib
import math

import numpy as np
import torch
from torch.nn import functional as F

WIDTH_MM = 64
SIZE = 224
PREFIX_BATCH = 16
IDENTITY = {'family': 'I', 'tx': 0.0, 'ty': 0.0, 'theta_deg': 0.0, 's': 1.0, 'sigma': 0.0, 'window': (1, 99), 'drop': None}


def seed_of(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:16], 16)


def view(name, family, **kw):
    v = dict(IDENTITY, family=family, **kw)
    v['name'] = name
    return v


def evaluation_views():
    """The 27 frozen evaluation views, in frozen order."""
    out = [view('I', 'I')]
    for d in (0.25, 0.5, 1.0):
        out.append(view('T_%g_col' % d, 'T', tx=d, dose=d))
        out.append(view('T_%g_row' % d, 'T', ty=d, dose=d))
    for deg in (2, 4, 6):
        for sign in (1, -1):
            out.append(view('R_%+d' % (sign * deg), 'R', theta_deg=float(sign * deg), dose=deg))
    for s in (0.97, 0.94, 0.90):
        out.append(view('S_%g' % s, 'S', s=s, dose=s))
    for s in (0.97, 0.94, 0.90):
        out.append(view('S_1/%g' % s, 'S', s=1 / s, dose=1 / s))
    for sigma in (0.02, 0.05, 0.10):
        out.append(view('N_%g' % sigma, 'N', sigma=sigma, dose=sigma))
    for lo, hi in ((2, 98), (5, 95), (0.5, 99.5)):
        out.append(view('W_%g_%g' % (lo, hi), 'W', window=(lo, hi)))
    out.append(view('D_center', 'D', drop='center'))
    out.append(view('D_edge', 'D', drop='edge'))
    assert len(out) == 27 and len({v['name'] for v in out}) == 27
    return out


def train_view(sid):
    """Fixed per-patient training view V: tx, ty ~ U[-1,1] px, theta ~ U[-4,4] deg, s ~ U[.94,1.06]."""
    rng = np.random.Generator(np.random.PCG64(seed_of('CS20261012|train_view|' + sid)))
    tx = float(rng.uniform(-1, 1)); ty = float(rng.uniform(-1, 1))
    theta = float(rng.uniform(-4, 4)); s = float(rng.uniform(.94, 1.06))
    return view('V', 'V', tx=tx, ty=ty, theta_deg=theta, s=s)


def noise_seed(sid, j):
    return seed_of('CS20261012|N|' + sid + '|' + str(j))


def check_dicom(ds):
    if str(ds.PhotometricInterpretation) not in ('MONOCHROME1', 'MONOCHROME2') or hasattr(ds, 'ModalityLUTSequence'):
        raise ValueError('unsupported DICOM')


def intensity(ds, window, sigma, seed):
    """Slice intensity pipeline: rescale -> percentile window on the whole slice -> clip -> MONOCHROME1 -> N."""
    image = ds.pixel_array.astype(np.float32)
    if image.ndim != 2:
        raise ValueError('unsupported DICOM')
    check_dicom(ds)
    image = image * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0)); assert np.isfinite(image).all()
    lo, hi = np.percentile(image, list(window))
    image = (np.zeros_like(image) if hi <= lo else np.clip((image - lo) / (hi - lo), 0, 1)).astype(np.float32)
    if hi > lo and str(ds.PhotometricInterpretation) == 'MONOCHROME1':
        image = 1 - image
    if sigma:
        z = np.random.Generator(np.random.PCG64(seed)).standard_normal(image.shape)
        image = np.clip(image + sigma * z, 0, 1).astype(np.float32)
    return image


def centre(ds, point, v):
    o = np.array(ds.ImageOrientationPatient, dtype=float); pos = np.array(ds.ImagePositionPatient, dtype=float); dy, dx = map(float, ds.PixelSpacing)
    x = float((point - pos) @ o[:3]) / dx; y = float((point - pos) @ o[3:]) / dy
    if v['tx']:
        x = x + v['tx']
    if v['ty']:
        y = y + v['ty']
    if not (0 <= x < int(ds.Columns) and 0 <= y < int(ds.Rows)):
        return None
    return x, y, dx, dy


def crop(image, x, y, dx, dy, v):
    width_mm = WIDTH_MM * v['s'] if v['s'] != 1 else WIDTH_MM
    offsets = (torch.arange(SIZE, dtype=torch.float32) + .5) / SIZE - .5
    if v['theta_deg'] == 0:
        xx = x + offsets * width_mm / dx; yy = y + offsets * width_mm / dy; gy, gx = torch.meshgrid(yy, xx, indexing='ij')
    else:
        # Positive theta rotates the column axis towards the row axis about the crop centre (mm).
        t = math.radians(v['theta_deg']); c, s = math.cos(t), math.sin(t)
        mm = offsets * width_mm; vv, uu = torch.meshgrid(mm, mm, indexing='ij')
        gx = x + (c * uu - s * vv) / dx; gy = y + (s * uu + c * vv) / dy
    h, w = image.shape
    grid = torch.stack((2 * (gx + .5) / w - 1, 2 * (gy + .5) / h - 1), -1)[None]
    out = F.grid_sample(torch.from_numpy(np.ascontiguousarray(image))[None, None], grid, align_corners=False, mode='bilinear', padding_mode='zeros')[0].repeat(3, 1, 1)
    return (out - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]


def drop_slice(mask, rule):
    """D family on the [5,3] slice mask: remove one visible slice per level (none if <=1 visible)."""
    out = mask.clone(); kept = 0
    for k in range(mask.shape[0]):
        vis = [j for j in range(mask.shape[1]) if bool(mask[k, j])]
        if len(vis) <= 1:
            kept += int(len(vis) == 1); continue
        j = vis[(len(vis) - 1) // 2] if rule == 'center' else vis[0]
        out[k, j] = False
    return out, kept


def load_prefix(weights, device='cuda'):
    from torchvision.models import resnet18
    backbone = resnet18(weights=None); backbone.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)); backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    def prefix(images):
        return backbone.layer3(backbone.layer2(backbone.layer1(backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images)))))))
    return backbone, prefix


def read_series(root, row):
    import pydicom
    assert len(row['input_files']) == 3
    dslist = [pydicom.dcmread(root / f) for f in row['input_files']]
    for ds in dslist:
        ori = np.array(ds.ImageOrientationPatient, dtype=float)
        assert str(ds.FrameOfReferenceUID) == row['frame'] and np.isfinite(np.r_[ori, np.array(ds.ImagePositionPatient), np.array(ds.PixelSpacing)]).all() and min(ds.PixelSpacing) > 0
        assert abs(np.linalg.norm(ori[:3]) - 1) < .001 and abs(np.linalg.norm(ori[3:]) - 1) < .001 and abs(ori[:3] @ ori[3:]) < .001
    return dslist


def level_points(row):
    o = np.array(row['center_orientation']); sp = row['center_pixel_spacing']; pos = np.array(row['center_position'])
    return [(k, pos + o[:3] * xy[0] * sp[1] + o[3:] * xy[1] * sp[0])
            for k, (xy, valid) in enumerate(zip(row['predicted_xy_original_pixels'], row['predicted_in_fov'])) if valid]


def encode(crops, locations, prefix, device):
    maps = torch.zeros(5, 3, 256, 14, 14); mask = torch.zeros(5, 3, dtype=torch.bool)
    if crops:
        with torch.inference_mode():
            encoded = torch.cat([prefix(torch.stack(crops[j:j + PREFIX_BATCH]).to(device)).cpu() for j in range(0, len(crops), PREFIX_BATCH)])
        assert encoded.shape[1:] == (256, 14, 14) and torch.isfinite(encoded).all()
        for location, feature in zip(locations, encoded):
            maps[location] = feature; mask[location] = True
    return maps, mask


def build_views(root, row, sid, views, prefix, device='cuda', timer=None):
    """{view name: (maps float32 [5,3,256,14,14], mask bool [5,3], info)} for one patient; one decode per slice."""
    dslist = read_series(root, row); points = level_points(row); images = {}; out = {}
    need_identity = any(v['family'] in ('I', 'D') for v in views)
    order = ([IDENTITY] if need_identity else []) + [v for v in views if v['family'] not in ('I', 'D')]
    for v in order:
        crops, locations = [], []
        for k, point in points:
            for j, ds in enumerate(dslist):
                c = centre(ds, point, v)
                if c is None:
                    continue
                key = (j, tuple(v['window']), v['sigma'])
                if key not in images:
                    images[key] = intensity(ds, v['window'], v['sigma'], noise_seed(sid, j))
                crops.append(crop(images[key], *c, v)); locations.append((k, j))
        maps, mask = encode(crops, locations, prefix, device)
        out['I' if v is IDENTITY else v['name']] = (maps, mask, {'crops': len(crops)})
    for v in views:
        if v['family'] == 'D':
            maps, mask, _ = out['I']; dmask, kept = drop_slice(mask, v['drop'])
            dmaps = maps.clone(); dmaps[mask & ~dmask] = 0
            out[v['name']] = (dmaps, dmask, {'levels_with_single_slice_not_dropped': kept})
    if not any(v['family'] == 'I' for v in views):
        out.pop('I', None)
    return out
