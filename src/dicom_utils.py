"""M0 engineering utilities. Whole-series bags; no coordinate labels in inference.

This is not the final level-localised diagnostic model or evidence policy.
Pixel scaling follows the documented legacy public_mri_data basics, implemented
here independently; existing experiments are not imported or restarted.
"""
import argparse,csv,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

LEVELS=['l1_l2','l2_l3','l3_l4','l4_l5','l5_s1']; GRADES={'Normal/Mild':0,'Moderate':1,'Severe':2}
def save_json(path,data):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels(path,size):
    ds=pydicom.dcmread(path);x=ds.pixel_array.astype(np.float32)
    if x.ndim!=2 or str(ds.PhotometricInterpretation) not in ('MONOCHROME1','MONOCHROME2'):raise ValueError('Unsupported image')
    if hasattr(ds,'ModalityLUTSequence'):raise ValueError('Unsupported Modality LUT')
    x=x*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
    if not np.isfinite(x).all():raise ValueError('Nonfinite pixels')
    lo,hi=np.percentile(x,[1,99]);constant=bool(hi<=lo)
    x=(np.zeros_like(x) if constant else np.clip((x-lo)/(hi-lo),0,1)).astype(np.float32)
    if not constant and str(ds.PhotometricInterpretation)=='MONOCHROME1':x=1-x
    h,w=x.shape;spacing=getattr(ds,'PixelSpacing',[1,1]);aspect=w*float(spacing[1])/(h*float(spacing[0]))
    nh=max(1,min(size,round(size/aspect)));nw=max(1,min(size,round(size*aspect)))
    if aspect>=1:nw=size
    else:nh=size
    x=F.interpolate(torch.from_numpy(np.ascontiguousarray(x))[None,None],size=(nh,nw),mode='bilinear',align_corners=False)[0]
    top=(size-nh)//2;left=(size-nw)//2;x=F.pad(x,(left,size-nw-left,top,size-nh-top))
    x=x.repeat(3,1,1);x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return x,constant
def scores(logits,y):
    p=logits.softmax(-1).cpu();pred=p.argmax(-1);mask=y>=0
    losses=F.cross_entropy(logits.cpu()[mask],y[mask],reduction='none')
    rows=torch.where(mask)[0];sums=torch.zeros(len(y)).scatter_add(0,rows,losses);counts=mask.sum(-1);usable=counts>0
    nll=float((sums[usable]/counts[usable]).mean().item())
    cm=np.zeros((3,3),dtype=int)
    for t,q in zip(y[mask].tolist(),pred[mask].tolist()):cm[t,q]+=1
    w=np.square(np.arange(3)[:,None]-np.arange(3)[None,:])/4;total=cm.sum();expected=np.outer(cm.sum(1),cm.sum(0))/total
    denom=float((w*expected).sum());qwk=None if denom==0 else float(1-(w*cm).sum()/denom)
    severe=y==2;complete=(y>=0).all(-1);patients=complete & severe.any(-1);miss=((severe)&(pred!=2)).any(-1)
    return {'nll':nll,'accuracy':float((pred[mask]==y[mask]).float().mean()),'qwk':qwk,'confusion':cm.tolist(),
        'complete_reference_patients':int(complete.sum()),'severe_patients':int(patients.sum()),'patient_severe_set_miss':None if not patients.any() else float(miss[patients].float().mean()),
        'severe_level_miss':None if not severe.any() else float((pred[severe]!=2).float().mean())}
def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);started=time.time();deadline=started+a.max_minutes*60
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    assert torch.cuda.is_available();device=torch.device('cuda');torch.backends.cuda.matmul.allow_tf32=False
    split=json.loads(Path(a.split).read_text(encoding='utf-8-sig'));root=Path(a.data_root)
    assert fingerprint(root/'train.csv')==split['train_csv_sha256']
    roles=split['patient_ids'];assert not(set(roles['training_pool'])&set(roles['model_selection']))
    selected={role:split['dicom_utils'][role] for role in ('training_pool','model_selection')}
    for role,ids in selected.items():assert len(set(ids))==len(ids) and set(ids)<=set(roles[role])
    assert not(set(sum(selected.values(),[]))&set(roles['source_confirmation']))
    labels={r['study_id']:torch.tensor([GRADES.get(r['spinal_canal_stenosis_'+l],-1) for l in LEVELS]) for r in csv.DictReader((root/'train.csv').open(encoding='utf-8-sig'))}
    series={}
    for r in csv.DictReader((root/'train_series_descriptions.csv').open(encoding='utf-8-sig')):
        series.setdefault(r['study_id'],[]).append((r['series_description'],r['series_id']))
    encoder=resnet18(weights=None);encoder.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));encoder.fc=nn.Identity();encoder.to(device).eval()
    for param in encoder.parameters():param.requires_grad_(False)
    spec={'phase':'M0','engineering_only':True,'variant':a.variant,'seed':a.seed,'image_size':a.image_size,'sagittal_limit':7,'axial_limit':15,
        'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'split_sha256':fingerprint(a.split),'source_sha256':fingerprint(__file__),
        'weights_sha256':fingerprint(a.weights),'coordinate_labels_used':False,'confirmation_or_external_used':False}
    save_json(out/'config.json',spec);features={};total_frames=0;constants=0;presence={};frame_count={};feature_start=time.time()
    for index,study in enumerate(sum(selected.values(),[])):
        if time.time()>deadline:raise TimeoutError('Wall-time exceeded')
        per=[];present=[];used=0
        for description,limit in [('Sagittal T2/STIR',7)]+([('Axial T2',15)] if a.variant=='sag_ax' else []):
            frames=[]
            for desc,sid in sorted(series[study],key=lambda t:(t[0],int(t[1]))):
                if desc!=description:continue
                paths=sorted((root/'train_images'/study/sid).glob('*.dcm'),key=lambda p:int(p.stem))
                if not paths:raise FileNotFoundError('Declared series empty')
                indices=np.linspace(0,len(paths)-1,min(len(paths),limit)).round().astype(int)
                for i in indices:
                    im,constant=pixels(paths[int(i)],a.image_size);frames.append(im);constants+=int(constant)
            if frames:
                with torch.inference_mode():
                    chunks=[encoder(torch.stack(frames[j:j+a.batch_size]).to(device)).cpu() for j in range(0,len(frames),a.batch_size)]
                per.append(torch.cat(chunks).mean(0));present.append(1.);used+=len(frames)
            else:per.append(torch.zeros(512));present.append(0.)
        features[study]=torch.cat(per+[torch.tensor(present)]);presence[study]=present;frame_count[study]=used;total_frames+=used
        if index%12==0:print(json.dumps({'event':'feature_progress','variant':a.variant,'patients':index+1,'frames':total_frames,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
    torch.cuda.synchronize();extract_seconds=time.time()-feature_start;encoder.cpu();del encoder;torch.cuda.empty_cache()
    torch.save({'features':features,'presence':presence,'frame_count':frame_count,'split_sha256':spec['split_sha256'],'config':spec},out/'feature_cache.pt')
    tx=torch.stack([features[s] for s in selected['training_pool']]).to(device);ty=torch.stack([labels[s] for s in selected['training_pool']]).to(device)
    vx=torch.stack([features[s] for s in selected['model_selection']]).to(device);vy=torch.stack([labels[s] for s in selected['model_selection']])
    mean=tx.mean(0);std=tx.std(0).clamp_min(.05);tx=(tx-mean)/std;vx=(vx-mean)/std
    head=nn.Sequential(nn.Linear(tx.shape[1],128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,15)).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);history=[];best=float('inf');best_epoch=0
    for epoch in range(a.epochs):
        if time.time()>deadline:raise TimeoutError('Wall-time exceeded')
        head.train();order=torch.randperm(len(tx),device=device);losses=[]
        for offset in range(0,len(order),16):
            ix=order[offset:offset+16];z=head(tx[ix]).reshape(-1,5,3);y=ty[ix];valid=y>=0
            if not valid.any():continue
            loss=F.cross_entropy(z[valid],y[valid]);assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5);optimizer.step();losses.append(float(loss.item()))
        head.eval()
        with torch.inference_mode():val=head(vx).reshape(-1,5,3);metrics=scores(val,vy)
        entry={'epoch':epoch+1,'train_nll':sum(losses)/len(losses),'selection':metrics};history.append(entry);print(json.dumps({'event':'epoch',**entry}),flush=True)
        if metrics['nll']<best:
            best=metrics['nll'];best_epoch=epoch+1
            torch.save({'state_dict':head.state_dict(),'mean':mean,'std':std,'input_dim':tx.shape[1],'config':spec,'best_epoch':best_epoch},out/'best_head.pt')
        save_json(out/'history.json',history)
    checkpoint=torch.load(out/'best_head.pt',map_location=device,weights_only=True);head.load_state_dict(checkpoint['state_dict']);head.eval()
    with torch.inference_mode():val=head(vx).reshape(-1,5,3);final=scores(val,vy);prob=val.softmax(-1).cpu().tolist()
    predictions=[{'study_id':s,'role':'model_selection','reference':labels[s].tolist(),'probabilities':p,'used_frames':frame_count[s],'presence':presence[s]} for s,p in zip(selected['model_selection'],prob)]
    save_json(out/'predictions.json',predictions)
    result={'status':'completed','engineering_only':True,'config':spec,'training_patients':len(tx),'selection_patients':len(vx),'decoded_frames':total_frames,'constant_frames':constants,
        'extract_seconds':extract_seconds,'frames_per_second':total_frames/extract_seconds,'elapsed_seconds':time.time()-started,
        'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'best_epoch':best_epoch,'selection':final,
        'quality_gate':{'decode_errors':0,'split_disjoint':True,'finite_training':True,'checkpoint_and_predictions_saved':True},
        'interpretation':'M0 pipeline and resource check only; not final level-localised model, policy efficacy, or external validation'}
    save_json(out/'result.json',result);print(json.dumps({'event':'completed','result':result}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--variant',choices=['sag','sag_ax'],required=True);p.add_argument('--seed',type=int,default=20261007);p.add_argument('--image-size',type=int,default=224);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--epochs',type=int,default=15);p.add_argument('--max-minutes',type=float,default=45)
    a=p.parse_args()
    try:run(a)
    except Exception as e:
        Path(a.output).mkdir(parents=True,exist_ok=True);save_json(Path(a.output)/'failure.json',{'status':'failed','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
