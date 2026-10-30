"""Predict all DEVELOPMENT roles without loading coordinate or grade CSVs."""
import argparse,csv,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch import nn
from torchvision.models import resnet18
from dicom_utils import pixels,save_json
def image_triplet(paths,size):
    raw=torch.stack([pixels(p,size)[0][0] for p in paths])*.229+.485
    return (raw-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
def header(path):
    ds=pydicom.dcmread(path,stop_before_pixels=True);o=np.array(ds.ImageOrientationPatient,dtype=float);pos=np.array(ds.ImagePositionPatient,dtype=float);sp=np.array(ds.PixelSpacing,dtype=float)
    if not np.isfinite(np.r_[o,pos,sp]).all() or min(sp)<=0:raise ValueError('invalid_geometry')
    if abs(np.linalg.norm(o[:3])-1)>.001 or abs(np.linalg.norm(o[3:])-1)>.001 or abs(o[:3]@o[3:])>.001:raise ValueError('invalid_directions')
    if str(getattr(ds,'AnatomicalOrientationType','BIPED'))!='BIPED':raise ValueError('non_biped')
    return {'path':path,'o':o,'pos':pos,'sp':sp,'n':np.cross(o[:3],o[3:]),'shape':[int(ds.Rows),int(ds.Columns)],'frame':str(getattr(ds,'FrameOfReferenceUID',''))}
def run(a):
    torch.set_num_threads(4);split=json.loads(Path(a.split).read_text(encoding='utf-8-sig'));root=Path(a.data_root);roles=split['patient_ids'];selected={s:r for r in ('diagnostic_train','policy_train','model_selection') for s in roles[r]};series=defaultdict(list)
    role_sets=[set(roles[r]) for r in roles];assert all(not (x&y) for i,x in enumerate(role_sets) for y in role_sets[i+1:]),'patient role overlap'
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert checkpoint['config']['split_sha256']==hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),'locator split mismatch'
    for row in csv.DictReader((root/'train_series_descriptions.csv').open(encoding='utf-8-sig')):
        if row['study_id'] in selected and row['series_description']=='Sagittal T2/STIR':series[row['study_id']].append(row['series_id'])
    model=resnet18(weights=None);model.fc=nn.Sequential(nn.Linear(512,10),nn.Sigmoid());model.load_state_dict(checkpoint['state_dict']);model.cuda().eval();size=checkpoint['config']['image_size'];rows=[];reject=[];start=time.time()
    for index,(study,role) in enumerate(selected.items()):
        try:
            choices=[(len(paths:=list((root/'train_images'/study/sid).glob('*.dcm'))),-int(sid),sid,paths) for sid in series[study]]
            if not choices:raise ValueError('missing_sagittal')
            _,_,sid,paths=max(choices,key=lambda x:(x[0],x[1]));heads=[header(p) for p in paths];normal=heads[0]['n'];heads.sort(key=lambda h:(float(h['pos']@normal),int(h['path'].stem)))
            if any(abs(float(h['n']@normal))<math.cos(math.radians(1)) for h in heads):raise ValueError('sagittal_orientation_inconsistent')
            if len({h['frame'] for h in heads})!=1 or not heads[0]['frame']:raise ValueError('inconsistent_or_unknown_series_frame')
            unique=[]
            for h in heads:
                if not unique or abs(float((h['pos']-unique[-1]['pos'])@normal))>.05:unique.append(h)
            mid=len(unique)//2;chosen=[unique[max(0,min(len(unique)-1,mid+i))] for i in (-1,0,1)];center=chosen[1]
            if len({h['path'] for h in chosen})<3:raise ValueError('fewer_than_3_unique_sagittal_slices')
            with torch.inference_mode():coords=model(image_triplet([h['path'] for h in chosen],size)[None].cuda()).reshape(5,2).cpu()
            h,w=center['shape'];dy,dx=center['sp'];aspect=w*dx/(h*dy);nh=max(1,min(size,round(size/aspect)));nw=max(1,min(size,round(size*aspect)))
            if aspect>=1:nw=size
            else:nh=size
            left=(size-nw)//2;top=(size-nh)//2;sx=nw/w;sy=nh/h;q=coords*size
            original=torch.stack(((q[:,0]-left-(sx-1)/2)/sx,(q[:,1]-top-(sy-1)/2)/sy),dim=-1)
            in_fov=((original[:,0]>=0)&(original[:,0]<w)&(original[:,1]>=0)&(original[:,1]<h)).tolist()
            rows.append({'study_id':study,'role':role,'series_id':sid,'center_file':center['path'].relative_to(root).as_posix(),'input_files':[h['path'].relative_to(root).as_posix() for h in chosen],
                'predicted_xy_original_pixels':original.tolist(),'predicted_in_fov':in_fov,'center_shape':center['shape'],'center_pixel_spacing':center['sp'].tolist(),'center_orientation':center['o'].tolist(),'center_position':center['pos'].tolist(),'frame':center['frame']})
        except Exception as e:reject.append({'study_id':study,'role':role,'reason':str(e)})
        if index%200==0:print(json.dumps({'event':'sagittal_predict','patients':index+1,'elapsed_seconds':round(time.time()-start,1)}),flush=True)
    result={'predictions':rows,'rejected':reject,'coordinate_csv_loaded':False,'grade_csv_loaded':False,'confirmation_or_external_used':False,'split_sha256':hashlib.sha256(Path(a.split).read_bytes()).hexdigest(),'locator_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);save_json(out/'predictions.json',result)
    role_sets=[set(roles[r]) for r in roles];assert all(not (x&y) for i,x in enumerate(role_sets) for y in role_sets[i+1:])
    unexpected=[r for r in reject if r['reason']!='missing_sagittal']
    summary={'status':'completed','engineering_only':True,'elapsed_seconds':time.time()-start,'predicted':len(rows),'rejected':reject,'roles':{r:sum(x['role']==r for x in rows) for r in ('diagnostic_train','policy_train','model_selection')},'quality_gate':{'decode_errors':len(unexpected),'split_disjoint':True,'finite_training':bool(rows) and all(np.isfinite(r['predicted_xy_original_pixels']).all() for r in rows),'checkpoint_and_predictions_saved':Path(a.checkpoint).exists() and (out/'predictions.json').exists()}}
    save_json(out/'result.json',summary);print(json.dumps(summary),flush=True)
    if unexpected or not summary['quality_gate']['finite_training']:
        summary['status']='failed';save_json(out/'result.json',summary);raise RuntimeError('Unexpected sagittal geometry/decode/inference failures; downstream blocked')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);run(p.parse_args())
