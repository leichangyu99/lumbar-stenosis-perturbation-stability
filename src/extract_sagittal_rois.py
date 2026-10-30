"""Frozen features from predicted sagittal ROIs, no diagnosis/coordinate CSV.
Within-series world geometry projects centre-plane predictions to each input.
Invalid/FOV-missing levels remain masked; no true-coordinate rescue.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from dicom_utils import save_json

def crop_image(ds,point,size,width_mm):
    o=np.array(ds.ImageOrientationPatient,dtype=float);pos=np.array(ds.ImagePositionPatient,dtype=float);dy,dx=map(float,ds.PixelSpacing)
    x=float((point-pos)@o[:3])/dx;y=float((point-pos)@o[3:])/dy
    if not (0<=x<int(ds.Columns) and 0<=y<int(ds.Rows)):return None
    image=ds.pixel_array.astype(np.float32)
    if image.ndim!=2 or str(ds.PhotometricInterpretation) not in ('MONOCHROME1','MONOCHROME2') or hasattr(ds,'ModalityLUTSequence'):raise ValueError('unsupported DICOM')
    image=image*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0));assert np.isfinite(image).all()
    lo,hi=np.percentile(image,[1,99]);image=(np.zeros_like(image) if hi<=lo else np.clip((image-lo)/(hi-lo),0,1)).astype(np.float32)
    if hi>lo and str(ds.PhotometricInterpretation)=='MONOCHROME1':image=1-image
    # Sample a square field in millimetres. grid_sample pads outside the FOV.
    offsets=(torch.arange(size,dtype=torch.float32)+.5)/size-.5
    xx=x+offsets*width_mm/dx;yy=y+offsets*width_mm/dy;gy,gx=torch.meshgrid(yy,xx,indexing='ij');h,w=image.shape
    grid=torch.stack((2*(gx+.5)/w-1,2*(gy+.5)/h-1),-1)[None]
    crop=F.grid_sample(torch.from_numpy(np.ascontiguousarray(image))[None,None],grid,align_corners=False,mode='bilinear',padding_mode='zeros')[0].repeat(3,1,1)
    return (crop-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]

def run(a):
    start=time.time();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    root=Path(a.data_root);split=json.loads(Path(a.split).read_text(encoding='utf-8-sig'));roles=split['patient_ids'];sets=[set(x) for x in roles.values()];assert all(not x&y for i,x in enumerate(sets) for y in sets[i+1:])
    preds=json.loads(Path(a.predictions).read_text(encoding='utf-8'));sha=hashlib.sha256(Path(a.split).read_bytes()).hexdigest();assert preds['split_sha256']==sha
    rows={r['study_id']:r for r in preds['predictions']};selected={s:r for r in ('diagnostic_train','policy_train','model_selection') for s in roles[r]}
    model=resnet18(weights=None);model.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));model.fc=nn.Identity();model.cuda().eval();features={};masks={};frame_counts={};nframes=0;decoded_frames=0;rejected=[]
    for index,(study,role) in enumerate(selected.items()):
        feature=torch.zeros(5,512);mask=torch.zeros(5,dtype=torch.bool);counts=torch.zeros(5,dtype=torch.int64);row=rows.get(study)
        if row:
            try:
                assert row['role']==role
                images=[pydicom.dcmread(root/f) for f in row['input_files']]
                for ds in images:
                    assert str(ds.FrameOfReferenceUID)==row['frame']
                    ori=np.array(ds.ImageOrientationPatient,dtype=float);assert np.isfinite(ori).all() and abs(np.linalg.norm(ori[:3])-1)<.001 and abs(np.linalg.norm(ori[3:])-1)<.001 and abs(ori[:3]@ori[3:])<.001
                o=np.array(row['center_orientation']);sp=row['center_pixel_spacing'];pos=np.array(row['center_position']);crops=[];indices=[];decoded=set()
                for k,(xy,valid) in enumerate(zip(row['predicted_xy_original_pixels'],row['predicted_in_fov'])):
                    if not valid:continue
                    point=pos+o[:3]*xy[0]*sp[1]+o[3:]*xy[1]*sp[0]
                    for image_index,ds in enumerate(images):
                        crop=crop_image(ds,point,a.image_size,a.width_mm)
                        if crop is not None:crops.append(crop);indices.append(k);decoded.add(image_index)
                if crops:
                    with torch.inference_mode():encoded=torch.cat([model(torch.stack(crops[j:j+a.batch_size]).cuda()).cpu() for j in range(0,len(crops),a.batch_size)])
                    for k in range(5):
                        ix=[j for j,i in enumerate(indices) if i==k]
                        if ix:feature[k]=encoded[ix].mean(0);mask[k]=True;counts[k]=len(ix)
                nframes+=len(images);decoded_frames+=len(decoded)
            except Exception as e:rejected.append({'study_id':study,'role':role,'reason':str(e)})
        features[study]=feature;masks[study]=mask;frame_counts[study]=counts
        if index%200==0:print(json.dumps({'event':'sag_roi_features','patients':index+1,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    config={'phase':'M1_sagittal_predicted_roi_features','split_sha256':sha,'predictions_sha256':hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),'weights_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'roi_width_mm':a.width_mm,'image_size':a.image_size,'coordinate_csv_loaded':False,'grade_csv_loaded':False,'confirmation_or_external_used':False}
    config['mask_semantics']='at least one input has predicted point within FOV; does not guarantee three slices or a complete 64mm window; boundaries zero padded'
    torch.save({'features':features,'masks':masks,'valid_slice_counts':frame_counts,'config':config},out/'feature_cache.pt')
    finite=all(bool(torch.isfinite(f).all()) for f in features.values())
    result={'status':'completed' if not rejected and finite else 'failed','engineering_only':True,'config':config,'patients':len(features),'fully_valid_five_roi_patients':sum(bool(m.all()) for m in masks.values()),'valid_level_rois':sum(int(m.sum()) for m in masks.values()),'read_input_frames':nframes,'decoded_unique_input_frames':decoded_frames,'rejected':rejected,'elapsed_seconds':time.time()-start,'quality_gate':{'decode_errors':len(rejected),'split_disjoint':True,'finite_training':finite,'checkpoint_and_predictions_saved':(out/'feature_cache.pt').exists() and Path(a.predictions).exists()}}
    save_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if rejected or not finite:raise RuntimeError('Unexpected sagittal ROI failures/nonfinite features; downstream blocked')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--predictions',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--image-size',type=int,default=224);p.add_argument('--width-mm',type=float,default=64);p.add_argument('--batch-size',type=int,default=16);run(p.parse_args())
