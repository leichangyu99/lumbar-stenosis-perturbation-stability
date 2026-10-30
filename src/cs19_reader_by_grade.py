"""Post hoc descriptive : reader A vs B disagreement stratified by the adjudicated reference grade (hospital 256)."""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cs13_hospital_stats as V  # noqa: E402
import cs13_hospital_labels as V2  # noqa: E402

OUT = HERE.parent.parent / 'results/clinical_addons_20261021/READER_BY_GRADE.json'


def main():
    _, _, _, _, ids = V.load_merged(); ref, A, B, _ = V2.read_labels(ids); k = (ref >= 0) & (A >= 0) & (B >= 0); res = {'note': 'post hoc descriptive; V1 only'}
    for g, n in ((0, 'normal_mild'), (1, 'moderate'), (2, 'severe')):
        m = k & (ref == g); res[n] = {'levels': int(m.sum()), 'disagreement': float((A[m] != B[m]).mean())}
    res['all'] = {'levels': int(k.sum()), 'disagreement': float((A[k] != B[k]).mean())}
    OUT.write_text(json.dumps(res, indent=1) + '\n', encoding='utf-8'); print(res)


if __name__ == '__main__':
    main()
