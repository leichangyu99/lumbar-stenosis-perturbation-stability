"""Geometry check of the SPIDER MetaImage->DICOM converter on expert masks (identity and oblique/permuted cases)."""
import re, sys, tempfile, zipfile
from pathlib import Path
import numpy as np, pydicom
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); import cs8_spider_convert as C
z = zipfile.ZipFile('data/public/SPIDER_masks_20261012/masks.zip')
names = [n for n in z.namelist() if re.fullmatch(r'masks/\d+_t2\.mha', n)]
pick = []
for n in names:
    h = z.read(n)[:800].decode(errors='ignore'); tm = re.search(r'TransformMatrix = (.*)', h).group(1)
    if tm.startswith('1 0 0') and len([p for p in pick if p[1]]) < 2: pick.append((n, True))
    elif not tm.startswith('1 0 0') and len([p for p in pick if not p[1]]) < 3: pick.append((n, False))
rng = np.random.default_rng(0)
for n, ident in pick:
    raw = z.read(n); arr, O, sp, D, _ = C.read_mha(raw); vol = np.transpose(arr, (2, 1, 0))
    d = Path(tempfile.mkdtemp()) / 'S'; info = C.convert(Path(n).stem, raw, d)
    files = sorted(d.glob('*.dcm')); dss = [pydicom.dcmread(f) for f in files]
    nz = np.argwhere(vol > 0); ok = 0
    for idx in nz[rng.integers(0, len(nz), 200)]:
        w = O + (idx * sp) @ D                                   # world = O + sum_a idx_a sp_a D[a]
        best = None
        for ds in dss:
            o = np.array(ds.ImageOrientationPatient, float); p = np.array(ds.ImagePositionPatient, float); dy, dx = map(float, ds.PixelSpacing)
            nrm = np.cross(o[:3], o[3:]); dist = abs((w - p) @ nrm)
            if best is None or dist < best[0]: best = (dist, ds, o, p, dy, dx)
        dist, ds, o, p, dy, dx = best; c = (w - p) @ o[:3] / dx; r = (w - p) @ o[3:] / dy
        ok += int(dist < 1e-3 and abs(c - round(c)) < 1e-3 and abs(r - round(r)) < 1e-3 and ds.pixel_array[int(round(r)), int(round(c))] == vol[tuple(idx)])
    print(n, 'identity' if ident else 'oblique/permuted', info, 'voxel matches', ok, '/ 200'); assert ok == 200
print('converter geometry passed')
