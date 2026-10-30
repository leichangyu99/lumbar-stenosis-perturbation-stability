# Perturbation stability of deep learning lumbar central canal stenosis grading

Code for the study "Small input perturbations change deep learning grades of lumbar central canal stenosis on MRI".

The code trains seven grader configurations (five backbones, three seeds each) for three-class central canal stenosis
grading on sagittal T2-weighted MRI, regenerates every input under 27 views (identity plus 26 computational
perturbations), and measures how often the predicted level grade changes (flip proportion). It also contains the
mitigation experiments (consistency training, AugMix, test-time augmentation, ensembling), the external hospital and
SPIDER evaluations, the conformal rule-out analysis, and the descriptive analyses reported in the manuscript.

No images, labels, patient-level predictions or model weights are included.

## Requirements

Python 3.10 and an NVIDIA GPU for training (inference and statistics also run on CPU).

```bash
pip install -r requirements.txt
```

## Data

| Dataset | Source | Expected location |
|---|---|---|
| RSNA 2024 Lumbar Spine Degenerative Classification (LumbarDISC) | Kaggle competition page, subject to the competition terms | `data/LumbarDISC/dataset` (`train.csv`, `train_images/`, series descriptions) |
| SPIDER lumbar MRI (CC BY 4.0) | Zenodo release of van der Graaf et al., Sci Data 2024 | `data/public/SPIDER_images_20261013/images.zip`, masks in `data/public/SPIDER_masks_20261012/masks.zip` |
| Hospital cohort | Not shared (de-identified clinical images) | `data/hospital/` |

Working files are written under `workspace/` (views, caches, trained models) and `results/` (statistics).

## Pretrained weights

ImageNet weights from torchvision (`resnet18-f37072fd.pth`, `resnet50-11ad3fa6.pth`, `convnext_tiny-983f1562.pth`,
`efficientnet_v2_s-dd5fe13b.pth`, `swin_t-704ceda3.pth`) and the semi-weakly supervised ResNet18
(`timm/resnet18.fb_swsl_ig1b_ft_in1k`, CC BY-NC 4.0) are verified by SHA256 before use (`src/cs3_arch.py`).
Place them in `workspace/weights/` (S0) and `weights_ma/` (A0–A5).

## Pipeline

| Step | Scripts | Manuscript |
|---|---|---|
| Canal-centre localisation (frozen locator) | `predict_sagittal.py`, `cs_locate.py` | Image processing |
| View engine: 27 views, ROI sampling | `cs_engine.py`, `cs_viewgen.py`, `extract_sagittal_rois.py` | Perturbations and endpoints |
| Flip and Jensen–Shannon metrics | `cs_metrics.py` | Endpoints |
| S0 grader and robust training (STD/AUG/CONS/SHUF) | `cs_common.py`, `cs_e2_train.py`, `train_sag_layer4_pair.py`, `train_sag_baseline.py` | Graders; Table 3 |
| Primary endpoint and robust training on test A | `cs_e34.py` | Results: stability; Table 3 |
| Sensitivity-guided active learning | `cs_e5_al.py` | ESM Table S5 |
| AugMix and test-time augmentation on test B | `cs2_views.py`, `cs2_engine.py`, `cs2_train.py`, `cs2_eval.py` | Table 3 |
| Multi-architecture graders A0–A5 | `cs3_arch.py`, `cs3_train.py`, `cs3_eval.py`, `cs11_val_predict.py` | Table 2; Fig. 3 |
| Conformal rule-out sets | `cs4_conformal.py`, `cs5_severe.py`, `cs6_ruleout.py`, `cs12_unified_conformal.py` | ESM Table S7 |
| Hospital cohort | `cs7_external_*.py`, `cs13_hospital_*.py` | Tables 1, 5; Fig. 6 |
| SPIDER | `cs8_spider_*.py` | Fig. 2; Fig. 3 |
| Ensembling | `cs14_mitigation.py`, `cs15_ensemble.py` | Table 4; Fig. 5 |
| Patient-level and descriptive analyses | `cs16_clinical.py`, `cs17_clinical.py`, `cs18_paper_panels.py`, `cs19_reader_by_grade.py` | Fig. 4; Fig. 5; ESM Tables S6, S11 |

Seeds are fixed in the scripts (20261012, 20261013, 20261014; bootstrap seeds as stated in each file). Prediction outputs are
recorded in `PREDICTIONS_LOCK.json`. Reference labels were completed before model predictions were generated. Docstrings refer to the
analysis protocols by section; the protocol documents are available from the corresponding author.

## Tests

Synthetic unit tests run without data:

```bash
PYTHONPATH=src python tests/test_detection_fixture.py
PYTHONPATH=src python tests/test_cs4_conformal_unit.py
PYTHONPATH=src python tests/test_cs5_unit.py
PYTHONPATH=src python tests/test_e34_stats_local.py
PYTHONPATH=src python tests/test_cs2_eval_stats_local.py
```

The remaining tests (`*_local.py`) need the pretrained weights or development DICOM files.

## License

MIT License (see `LICENSE`). Third-party weights and datasets keep their own licenses.

## Contact

Ming Chen (651530536@qq.com) and Hui Kang (kanghuispine@163.com).
