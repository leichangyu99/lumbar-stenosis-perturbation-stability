"""Bounded, matched frozen versus layer4-adapted sagittal diagnosis.

Fixed predicted ROIs and grayscale pooling; no axial or coordinate labels.
Caches the frozen prefix through layer3, NOT final 512-dimensional features.
All preprocessing, both controls and OOM retry share one stage deadline.
"""
import argparse,copy,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import pydicom
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18
from dicom_utils import LEVELS,GRADES,scores,save_json
from extract_sagittal_rois import crop_image
from train_sag_baseline import SagHead
SEED=20261007

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def bn_state(module):return {n:t.detach().clone() for n,t in module.state_dict().items() if n.endswith(('running_mean','running_var','num_batches_tracked'))}
def same_state(a,b):return a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a)

class Diagnosis(nn.Module):
    def __init__(self,layer4,head,mean,std,adapt):
        super().__init__();self.layer4=copy.deepcopy(layer4);self.head=copy.deepcopy(head);self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        for p in self.layer4.parameters():p.requires_grad_(adapt)
    def train(self,mode=True):
        super().train(mode);self.layer4.eval();return self
    def encode(self,maps,mask):
        # layer4 deliberately remains outside no_grad during training.
        per=torch.zeros((*mask.shape,512),device=maps.device)
        if mask.any():per[mask]=F.adaptive_avg_pool2d(self.layer4(maps[mask]),1).flatten(1)
        count=mask.sum(-1);return per.sum(2)/count.clamp_min(1)[...,None],count>0
    def forward(self,maps,mask):
        features,valid=self.encode(maps,mask);return self.head((features-self.mean)/self.std,valid)

def extended_scores(z,y):
    result=scores(z,y);cm=np.array(result['confusion']);recall=np.divide(cm.diagonal(),cm.sum(1),out=np.zeros(3,dtype=float),where=cm.sum(1)>0);result['per_class_recall']=recall.tolist();result['balanced_accuracy']=float(recall[cm.sum(1)>0].mean())
    known=y>=0;p=z.softmax(-1).detach().cpu()[known];target=y[known];confidence,guess=p.max(-1);correct=(guess==target).float();ece=0.
    for k in range(15):
        take=(confidence>=k/15)&((confidence<(k+1)/15) if k<14 else (confidence<=1))
        if take.any():ece+=float(take.float().mean()*abs(confidence[take].mean()-correct[take].mean()))
    result['ece_15_bins_level_weighted']=ece;return result

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);stage=out.parent;clockfile=stage/'stage_clock.json'
    if not clockfile.exists():save_json(clockfile,{'started_epoch':time.time(),'total_max_seconds':a.max_minutes*60})
    clock=json.loads(clockfile.read_text());assert clock['total_max_seconds']==a.max_minutes*60;deadline=clock['started_epoch']+clock['total_max_seconds'];started=time.time()
    def check():
        if time.time()>=deadline:raise TimeoutError('Whole paired stage including preprocessing/retry exceeded deadline')
    check();torch.set_num_threads(4);torch.manual_seed(SEED);np.random.seed(SEED);assert torch.cuda.is_available();assert 1<=a.epochs<=12 and a.batch_size>=1
    root=Path(a.data_root);split=json.loads(Path(a.split).read_text(encoding='utf-8-sig'));roles=split['patient_ids'];sets=[set(x) for x in roles.values()];assert all(not x&y for i,x in enumerate(sets) for y in sets[i+1:]);parent=json.loads(Path(a.parent_split).read_text(encoding='utf-8-sig'));assert split['parent_split_sha256']==sha(a.parent_split);assert parent['train_csv_sha256']==sha(root/'train.csv')
    ids={r:roles[r] for r in ('diagnostic_train','model_selection')};selected={s:r for r in ids for s in ids[r]};pred=json.loads(Path(a.predictions).read_text(encoding='utf-8'));assert pred['split_sha256']==sha(a.split);rows={r['study_id']:r for r in pred['predictions']};expected_missing={r['study_id'] for r in pred['rejected'] if r['reason']=='missing_sagittal'}
    backbone=resnet18(weights=None);backbone.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True));backbone.cuda().eval()
    for p in backbone.parameters():p.requires_grad_(False)
    prefix_state={n:t for n,t in backbone.state_dict().items() if not n.startswith(('layer4.','fc.'))};prefix_before=state_hash(prefix_state)
    def prefix(images):return backbone.layer3(backbone.layer2(backbone.layer1(backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(images)))))))
    cache=stage/'front_cache';cache.mkdir(exist_ok=True);spec={'split_sha256':sha(a.split),'prediction_sha256':sha(a.predictions),'weights_sha256':sha(a.weights),'roi_helper_sha256':sha(Path(__file__).with_name('extract_sagittal_rois.py')),'source_sha256':sha(__file__),'roi_width_mm':64,'image_size':224,'cache_tensor':'float32 layer3 maps [5,3,256,14,14]','roles':list(ids),'coordinate_csv_loaded':False,'axial_images_loaded':False,'policy_train_used':False,'confirmation_or_external_used':False}
    cache_meta=cache/'spec.json'
    if cache_meta.exists():assert json.loads(cache_meta.read_text())==spec,'cache spec mismatch'
    else:save_json(cache_meta,spec)
    decoded=0;reused=0
    for index,(study,role) in enumerate(selected.items()):
        check();path=cache/(study+'.pt')
        if path.exists():reused+=1;continue
        maps=torch.zeros(5,3,256,14,14);mask=torch.zeros(5,3,dtype=torch.bool);row=rows.get(study)
        if row is None:assert study in expected_missing,'unexplained missing prediction'
        else:
            assert row['role']==role and len(row['input_files'])==3
            dslist=[pydicom.dcmread(root/f) for f in row['input_files']];o=np.array(row['center_orientation']);sp=row['center_pixel_spacing'];pos=np.array(row['center_position']);crops=[];locations=[];decoded_indices=set()
            for ds in dslist:
                ori=np.array(ds.ImageOrientationPatient,dtype=float);assert str(ds.FrameOfReferenceUID)==row['frame'] and np.isfinite(np.r_[ori,np.array(ds.ImagePositionPatient),np.array(ds.PixelSpacing)]).all() and min(ds.PixelSpacing)>0
                assert abs(np.linalg.norm(ori[:3])-1)<.001 and abs(np.linalg.norm(ori[3:])-1)<.001 and abs(ori[:3]@ori[3:])<.001
            for k,(xy,valid) in enumerate(zip(row['predicted_xy_original_pixels'],row['predicted_in_fov'])):
                if not valid:continue
                point=pos+o[:3]*xy[0]*sp[1]+o[3:]*xy[1]*sp[0]
                for j,ds in enumerate(dslist):
                    image=crop_image(ds,point,224,64)
                    if image is not None:crops.append(image);locations.append((k,j));decoded_indices.add(j)
            if crops:
                with torch.inference_mode():encoded=torch.cat([prefix(torch.stack(crops[j:j+16]).cuda()).cpu() for j in range(0,len(crops),16)])
                assert encoded.shape[1:]==(256,14,14) and torch.isfinite(encoded).all()
                for location,feature in zip(locations,encoded):maps[location]=feature;mask[location]=True
            decoded+=len(decoded_indices)
        # Atomic per-patient caches allow a bounded OOM retry to reuse inputs.
        temp=path.with_suffix('.tmp');torch.save({'maps':maps,'mask':mask,'study_id':study,'role':role},temp);temp.replace(path)
        if index%100==0:print(json.dumps({'event':'prefix_cache','patients':index+1,'elapsed_seconds':round(time.time()-started,1)}),flush=True)
    assert prefix_before==state_hash(prefix_state),'frozen prefix/BN changed during preprocessing';save_json(cache/'ready.json',{'status':'completed','patients':len(selected),'spec':spec,'decoded_this_attempt':decoded,'reused_patient_caches':reused})
    def load_batch(studies):
        records=[torch.load(cache/(s+'.pt'),map_location='cpu',weights_only=True) for s in studies]
        assert all(r['study_id']==s and r['role']==selected[s] for r,s in zip(records,studies))
        return torch.stack([r['maps'] for r in records]).cuda(),torch.stack([r['mask'] for r in records]).cuda()
    # Initial 512-dimensional statistics use diagnostic_train only, and stay fixed.
    frozen_features=[];valid_features=[]
    for offset in range(0,len(ids['diagnostic_train']),a.batch_size):
        check();maps,mask=load_batch(ids['diagnostic_train'][offset:offset+a.batch_size]);per=torch.zeros((*mask.shape,512),device='cuda')
        with torch.no_grad():
            if mask.any():per[mask]=F.adaptive_avg_pool2d(backbone.layer4(maps[mask]),1).flatten(1)
        count=mask.sum(-1);feature=per.sum(2)/count.clamp_min(1)[...,None];frozen_features.append(feature.cpu());valid_features.append((count>0).cpu())
    all_features=torch.cat(frozen_features);all_valid=torch.cat(valid_features);valid=all_features[all_valid];assert len(valid)>0;mean=valid.mean(0).cuda();std=valid.std(0).clamp_min(.05).cuda()
    labels={}
    for row in csv.DictReader((root/'train.csv').open(encoding='utf-8-sig')):
        if row['study_id'] in selected:labels[row['study_id']]=torch.tensor([GRADES.get(row['spinal_canal_stenosis_'+level],-1) for level in LEVELS])
    assert labels.keys()==selected.keys();ty=torch.stack([labels[s] for s in ids['diagnostic_train']]);vy=torch.stack([labels[s] for s in ids['model_selection']]);counts=torch.bincount(ty[ty>=0],minlength=3).float().cuda();assert (counts>0).all();weights=(counts.sum()/(3*counts)).sqrt();weights/=weights.mean()
    torch.manual_seed(SEED);head=SagHead().cuda();template=Diagnosis(backbone.layer4,head,mean,std,True).cuda()
    # Witness uses a throwaway copy: warm-up cannot alter either experimental branch.
    witness=copy.deepcopy(template);witness.train();wmaps,wmask=load_batch(ids['diagnostic_train'][:min(2,a.batch_size)]);target=ty[:len(wmaps)].cuda();known=target>=0;before_l4=state_hash(witness.layer4.state_dict());before_head=state_hash(witness.head.state_dict());before_bn=bn_state(witness.layer4)
    opt=torch.optim.AdamW(witness.parameters(),lr=1e-4);logits=witness(wmaps,wmask);loss=F.cross_entropy(logits[known],target[known]);loss.backward();l4_grad=sum(float(p.grad.abs().sum()) for p in witness.layer4.parameters() if p.grad is not None);head_grad=sum(float(p.grad.abs().sum()) for p in witness.head.parameters() if p.grad is not None);assert np.isfinite([l4_grad,head_grad]).all() and l4_grad>0 and head_grad>0;opt.step()
    audit={'real_diagnostic_patients':ids['diagnostic_train'][:len(wmaps)],'layer4_finite_nonzero_gradient':l4_grad>0,'head_finite_nonzero_gradient':head_grad>0,'layer4_weights_changed':before_l4!=state_hash(witness.layer4.state_dict()),'head_weights_changed':before_head!=state_hash(witness.head.state_dict()),'layer4_bn_statistics_unchanged':same_state(before_bn,bn_state(witness.layer4)),'frozen_prefix_bn_and_weights_unchanged':prefix_before==state_hash(prefix_state),'prefix_has_no_grad':all(p.grad is None and not p.requires_grad for n,p in backbone.named_parameters()),'cache_before_layer4':True,'witness_weights_discarded':True};assert all(v is True for k,v in audit.items() if k!='real_diagnostic_patients');save_json(out/'gradient_witness.json',audit);del witness,opt,wmaps,wmask,logits;torch.cuda.empty_cache()
    config={'phase':'M1_matched_sag_layer4_adaptation','spec':spec,'seed':SEED,'epochs':a.epochs,'microbatch_patients':a.batch_size,'effective_patient_batch':32,'layer4_lr':1e-4,'head_lr':1e-3,'weight_decay':.01,'bn_statistics':'frozen all','loss':'patient-equal masked CE with diagnostic sqrt inverse-frequency weights','class_weights':weights.cpu().tolist(),'checkpoint_selection':'unweighted patient-equal model_selection NLL only','stage_total_wall_limit_seconds':clock['total_max_seconds'],'not_innovation_or_clinical_efficacy':True};save_json(out/'config.json',config)
    def evaluate(net):
        net.eval();logits=[];masks=[];slice_counts=[]
        with torch.no_grad():
            for offset in range(0,len(vy),a.batch_size):
                check();maps,mask=load_batch(ids['model_selection'][offset:offset+a.batch_size]);logits.append(net(maps,mask).cpu());masks.append(mask.any(-1).cpu());slice_counts.append(mask.sum(-1).cpu())
        z=torch.cat(logits);assert torch.isfinite(z).all();return extended_scores(z,vy),z,torch.cat(masks),torch.cat(slice_counts)
    # Identical initial outputs confirm the paired comparison isolates adaptation.
    m0,z0,_,_=evaluate(template);save_json(out/'epoch0.json',{'metrics':m0,'head_hash':state_hash(template.head.state_dict()),'layer4_hash':state_hash(template.layer4.state_dict())})
    variants={};all_predictions={}
    for name,adapt in [('frozen',False),('layer4',True)]:
        check();torch.manual_seed(SEED);net=copy.deepcopy(template)
        for p in net.layer4.parameters():p.requires_grad_(adapt)
        groups=[{'params':list(net.head.parameters()),'lr':1e-3}]
        if adapt:groups.append({'params':list(net.layer4.parameters()),'lr':1e-4})
        optimizer=torch.optim.AdamW(groups,weight_decay=.01);initial_l4=state_hash(net.layer4.state_dict());initial_bn=bn_state(net.layer4);initial_head=state_hash(net.head.state_dict());directory=out/name;directory.mkdir(exist_ok=True);history=[];best=float('inf');variant_start=time.time();orders=[]
        init_metrics,initial_logits,_,_=evaluate(net);assert torch.equal(z0,initial_logits),'branch initialization mismatch'
        for epoch in range(a.epochs):
            check();net.train();order=torch.randperm(len(ty),device='cuda').cpu().tolist();orders.append(hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest());losses=[]
            for offset in range(0,len(order),32):
                check();batch=order[offset:offset+32];valid_count=int((ty[batch]>=0).any(-1).sum())
                if not valid_count:continue
                optimizer.zero_grad();batch_loss=0.
                for start in range(0,len(batch),a.batch_size):
                    ix=batch[start:start+a.batch_size];maps,mask=load_batch([ids['diagnostic_train'][i] for i in ix]);target=ty[ix].cuda();known=target>=0;z=net(maps,mask);per_level=F.cross_entropy(z.reshape(-1,3),target.clamp_min(0).reshape(-1),weight=weights,reduction='none').reshape(-1,5)*known;usable=known.sum(1)>0
                    if usable.any():
                        loss=(per_level.sum(1)[usable]/known.sum(1)[usable]).sum()/valid_count;assert torch.isfinite(loss);loss.backward();batch_loss+=float(loss.detach())
                nn.utils.clip_grad_norm_(net.head.parameters(),5)
                if adapt:nn.utils.clip_grad_norm_(net.layer4.parameters(),5)
                optimizer.step();losses.append(batch_loss)
            metrics,_,_,_=evaluate(net);entry={'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'selection':metrics};history.append(entry);save_json(directory/'history.json',history);print(json.dumps({'event':'paired_epoch','variant':name,**entry}),flush=True)
            if metrics['nll']<best:
                best=metrics['nll'];torch.save({'state_dict':net.state_dict(),'epoch':epoch+1,'config':config,'variant':name},directory/'best.pt')
        changed=initial_l4!=state_hash(net.layer4.state_dict());assert changed==adapt and same_state(initial_bn,bn_state(net.layer4)) and prefix_before==state_hash(prefix_state),'frozen state changed or encoder failed to adapt'
        checkpoint=torch.load(directory/'best.pt',map_location='cuda',weights_only=True);net.load_state_dict(checkpoint['state_dict']);metrics,z,mask,counts=evaluate(net);prob=z.softmax(-1).tolist();predictions=[{'study_id':s,'reference':labels[s].tolist(),'probabilities':p,'sag_roi_mask':m,'valid_slice_counts':c} for s,p,m,c in zip(ids['model_selection'],prob,mask.tolist(),counts.tolist())];save_json(directory/'predictions.json',predictions);all_predictions[name]=predictions
        variants[name]={'selection':metrics,'best_epoch':checkpoint['epoch'],'elapsed_seconds':time.time()-variant_start,'initial_head_hash':initial_head,'initial_layer4_hash':initial_l4,'layer4_changed':changed,'bn_statistics_unchanged':True,'order_hashes':orders};save_json(directory/'result.json',variants[name]);del net,optimizer;torch.cuda.empty_cache()
    assert variants['frozen']['initial_head_hash']==variants['layer4']['initial_head_hash'] and variants['frozen']['order_hashes']==variants['layer4']['order_hashes'],'unmatched initialization or patient order'
    save_json(out/'predictions.json',all_predictions);result={'status':'completed','engineering_only':True,'config':config,'variants':variants,'diagnostic_train_patients':len(ty),'model_selection_patients':len(vy),'missing_sagittal_patients':sorted(set(selected)&expected_missing),'total_stage_seconds':time.time()-clock['started_epoch'],'elapsed_seconds_this_attempt':time.time()-started,'peak_cuda_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'quality_gate':{'decode_errors':0,'split_disjoint':True,'finite_training':True,'checkpoint_and_predictions_saved':all((out/name/'best.pt').exists() for name in variants) and (out/'predictions.json').exists()},'interpretation':'Bounded matched public development diagnostic representation experiment; no accurate axial packages, policy efficacy, independent confirmation or clinical safety claim.'};save_json(out/'result.json',result);print(json.dumps({'event':'completed','result':result}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--parent-split',required=True);p.add_argument('--predictions',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--max-minutes',type=int,default=119);a=p.parse_args()
    try:run(a)
    except Exception as e:
        import traceback
        Path(a.output).mkdir(parents=True,exist_ok=True);save_json(Path(a.output)/'failure.json',{'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
