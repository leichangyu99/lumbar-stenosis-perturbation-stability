"""CS20261013-SPX: convert SPIDER T2 sagittal MetaImage volumes (public, CC-BY 4.0) into per-slice DICOM series with exact
geometry so the frozen locator / intervention engine can be reused unchanged.  Direction rows of TransformMatrix are the
image-axis direction cosines (ITK MetaImageIO convention).  The slice axis is the one most aligned with L-R; in-plane
axes are flipped as needed so image rows run anterior->posterior (+P) and columns superior->inferior (-S), as in RSNA."""
import hashlib, json, re, sys, zipfile, zlib
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

SRC = Path('data/public/SPIDER_images_20261013/images.zip')
OUT = Path('data/public/SPIDER_dicom_20261013')
DT = {'MET_SHORT': np.int16, 'MET_USHORT': np.uint16, 'MET_UCHAR': np.uint8, 'MET_CHAR': np.int8, 'MET_INT': np.int32, 'MET_UINT': np.uint32, 'MET_FLOAT': np.float32}


def read_mha(raw):
    end = raw.find(b'ElementDataFile'); end = raw.find(b'\n', end) + 1
    h = dict(l.split(' = ', 1) for l in raw[:end].decode().splitlines() if ' = ' in l)
    assert h['ElementDataFile'].strip() == 'LOCAL' and h['BinaryDataByteOrderMSB'].strip() == 'False'
    data = raw[end:]; data = zlib.decompress(data) if h.get('CompressedData', 'False').strip() == 'True' else data
    dims = [int(x) for x in h['DimSize'].split()]; arr = np.frombuffer(data, dtype=DT[h['ElementType'].strip()]).reshape(dims[::-1])   # [k][j][i]
    D = np.array([float(x) for x in h['TransformMatrix'].split()]).reshape(3, 3)   # row a = direction of axis a
    return arr, np.array([float(x) for x in h['Offset'].split()]), np.array([float(x) for x in h['ElementSpacing'].split()]), D, h


def uid(text): return '2.25.' + str(int(hashlib.sha256(('CS20261013|SPIDER|' + text).encode()).hexdigest()[:30], 16))


def convert(name, raw, out_dir):
    arr, O, sp, D, h = read_mha(raw); vol = np.transpose(arr, (2, 1, 0))          # vol[i,j,k]
    s = int(np.argmax(np.abs(D[:, 0]))); rest = [a for a in range(3) if a != s]
    a = max(rest, key=lambda q: abs(D[q, 1])); b = [q for q in rest if q != a][0]   # a ~ AP (image columns), b ~ SI (image rows)
    assert abs(D[s, 0]) > .9 and abs(D[a, 1]) > .9 and abs(D[b, 2]) > .9, 'not a sagittal acquisition'
    flip_a = D[a, 1] < 0; flip_b = D[b, 2] > 0                                         # want +P along columns, -S down rows
    row_dir = D[a] * (-1 if flip_a else 1); col_dir = D[b] * (-1 if flip_b else 1)
    v = np.moveaxis(vol, (s, b, a), (0, 1, 2))                                         # [slice, row(b), col(a)]
    if flip_a: v = v[:, :, ::-1]
    if flip_b: v = v[:, ::-1, :]
    Na, Nb = vol.shape[a], vol.shape[b]
    study, series, frame = uid(name + '|study'), uid(name + '|series'), uid(name + '|frame')
    signed = v.dtype.kind == 'i'; assert v.dtype.itemsize <= 2, 'pixel type'
    out_dir.mkdir(parents=True)
    for t in range(v.shape[0]):
        pos = O + t * sp[s] * D[s] + (Na - 1) * sp[a] * D[a] * flip_a + (Nb - 1) * sp[b] * D[b] * flip_b
        ds = Dataset(); fm = FileMetaDataset(); fm.MediaStorageSOPClassUID = MRImageStorage; fm.MediaStorageSOPInstanceUID = uid('%s|%d' % (name, t)); fm.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta = fm; ds.SOPClassUID = MRImageStorage; ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
        ds.Modality = 'MR'; ds.PatientName = 'SPIDER_' + name; ds.PatientID = 'SPIDER_' + name; ds.StudyInstanceUID = study; ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = frame; ds.SeriesDescription = 'SPIDER T2 sagittal (converted from MetaImage)'; ds.InstanceNumber = t + 1
        ds.ImageOrientationPatient = [float(x) for x in np.r_[row_dir, col_dir]]; ds.ImagePositionPatient = [float(x) for x in pos]
        ds.PixelSpacing = [float(sp[b]), float(sp[a])]; ds.SliceThickness = float(sp[s]); ds.Rows, ds.Columns = int(Nb), int(Na)
        ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'; ds.BitsAllocated = 16; ds.BitsStored = 16; ds.HighBit = 15
        ds.PixelRepresentation = 1 if signed else 0
        img = np.ascontiguousarray(v[t]).astype(np.int16 if signed else np.uint16); ds.PixelData = img.tobytes()
        ds.save_as(out_dir / ('%04d.dcm' % (t + 1)), enforce_file_format=True)
    return {'slices': int(v.shape[0]), 'slice_axis': s, 'flip_ap': bool(flip_a), 'flip_si': bool(flip_b), 'thickness_mm': float(sp[s]),
            'inplane_mm': [float(sp[b]), float(sp[a])], 'rows_cols': [int(Nb), int(Na)], 'oblique_deg': float(np.degrees(np.arccos(min(1, abs(D[s, 0])))))}


def main(limit=None):
    z = zipfile.ZipFile(SRC); names = sorted(n for n in z.namelist() if re.fullmatch(r'images/\d+_t2\.mha', n))
    OUT.mkdir(exist_ok=limit is not None); rep = {}
    for n in names[:limit]:
        case = 'SP-' + Path(n).stem.split('_')[0].zfill(3)
        try: rep[case] = convert(Path(n).stem, z.read(n), OUT / case / 'SAG_T2')
        except Exception as e: rep[case] = {'error': str(e)}
    (OUT / 'CONVERT_REPORT.json').write_text(json.dumps(rep, indent=1), encoding='utf-8')
    print(len(names), sum('error' not in r for r in rep.values()), [k for k, r in rep.items() if 'error' in r][:10])


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
