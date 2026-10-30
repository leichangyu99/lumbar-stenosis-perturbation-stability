"""M1 predicted sagittal ROI diagnostic feasibility baseline, not a policy.

Only diagnostic_train grades/ features fit parameters and normalisation. This
separate exploratory head is not the later fixed multimodal policy diagnosis.
"""
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from dicom_utils import LEVELS,GRADES,scores,save_json

class SagHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(1028,192),nn.ReLU(),nn.Dropout(.1),nn.Linear(192,3))
    def forward(self,x,mask):
        visible=x*mask[...,None];context=visible.sum(1)/mask.sum(1).clamp_min(1)[:,None]
        levels=torch.linspace(0,1,5,device=x.device)[None,:,None].expand(len(x),-1,-1)
        position=torch.cat((levels,levels.square(),mask[...,None].float(),mask.float().mean(1)[:,None,None].expand(-1,5,-1)),dim=-1)
        return self.net(torch.cat((visible,context[:,None].expand(-1,5,-1),position),dim=-1))

def run(a):
    started=time.time();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);torch.manual_seed(20261007)
    split=json.loads(Path(a.split).read_text(encoding='utf-8-sig'));roles=split['patient_ids'];sets=[set(x) for x in roles.values()];assert all(not x&y for i,x in enumerate(sets) for y in sets[i+1:]);cache=torch.load(a.features,map_location='cpu',weights_only=True);assert cache['config']['split_sha256']==hashlib.sha256(Path(a.split).read_bytes()).hexdigest()
    ids={r:roles[r] for r in ('diagnostic_train','model_selection')};allowed=set(ids['diagnostic_train']+ids['model_selection']);labels={}
    for row in csv.DictReader((Path(a.data_root)/'train.csv').open(encoding='utf-8-sig')):
        if row['study_id'] in allowed:labels[row['study_id']]=torch.tensor([GRADES.get(row['spinal_canal_stenosis_'+level],-1) for level in LEVELS])
    x={r:torch.stack([cache['features'][s] for s in ids[r]]).cuda() for r in ids};m={r:torch.stack([cache['masks'][s] for s in ids[r]]).cuda() for r in ids};y={r:torch.stack([labels[s] for s in ids[r]]) for r in ids};ty=y['diagnostic_train'].cuda();valid=m['diagnostic_train'];mean=x['diagnostic_train'][valid].mean(0);std=x['diagnostic_train'][valid].std(0).clamp_min(.05)
    for role in x:x[role]=(x[role]-mean)/std
    counts=torch.bincount(ty[ty>=0],minlength=3).float();weights=(counts.sum()/(3*counts.clamp_min(1))).sqrt();weights/=weights.mean()
    head=SagHead().cuda();opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);history=[];best=float('inf')
    config={'phase':'M1_sag_predicted_roi_diagnostic_feasibility','split_sha256':cache['config']['split_sha256'],'feature_config':cache['config'],'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training':'patient-equal masked CE with square-root inverse-frequency weights','class_weights':weights.cpu().tolist(),'diagnostic_train_patients':len(ids['diagnostic_train']),'policy_train_used':False,'confirmation_or_external_used':False,'not_final_policy_diagnostic_head':True};save_json(out/'config.json',config)
    for epoch in range(a.epochs):
        head.train();order=torch.randperm(len(ty),device='cuda');losses=[]
        for offset in range(0,len(order),32):
            ix=order[offset:offset+32];logits=head(x['diagnostic_train'][ix],m['diagnostic_train'][ix]);target=ty[ix];mask=target>=0;losses_per_level=F.cross_entropy(logits.reshape(-1,3),target.clamp_min(0).reshape(-1),weight=weights,reduction='none').reshape(-1,5)*mask;usable=mask.sum(1)>0
            if not usable.any():continue
            loss=(losses_per_level.sum(1)[usable]/mask.sum(1)[usable]).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5);opt.step();losses.append(float(loss))
        head.eval()
        with torch.inference_mode():logits=head(x['model_selection'],m['model_selection']);metrics=scores(logits,y['model_selection'])
        entry={'epoch':epoch+1,'train_loss':sum(losses)/len(losses),'selection':metrics};history.append(entry);save_json(out/'history.json',history);print(json.dumps(entry),flush=True)
        if metrics['nll']<best:
            best=metrics['nll'];torch.save({'state_dict':head.state_dict(),'mean':mean,'std':std,'epoch':epoch+1,'config':config},out/'best_head.pt')
    checkpoint=torch.load(out/'best_head.pt',map_location='cuda',weights_only=True);head.load_state_dict(checkpoint['state_dict']);head.eval()
    with torch.inference_mode():logits=head(x['model_selection'],m['model_selection']);final=scores(logits,y['model_selection']);prob=logits.softmax(-1).cpu().tolist()
    predictions=[{'study_id':s,'reference':labels[s].tolist(),'probabilities':p,'sag_roi_mask':cache['masks'][s].tolist(),'valid_slice_counts':cache['valid_slice_counts'][s].tolist()} for s,p in zip(ids['model_selection'],prob)];save_json(out/'predictions.json',predictions)
    result={'status':'completed','engineering_only':True,'config':config,'selection':final,'best_epoch':checkpoint['epoch'],'elapsed_seconds':time.time()-started,'quality_gate':{'decode_errors':0,'split_disjoint':True,'finite_training':bool(torch.isfinite(logits).all()),'checkpoint_and_predictions_saved':(out/'best_head.pt').exists() and (out/'predictions.json').exists()},'interpretation':'Predicted sagittal ROI feasibility; no axial benefit, policy, external validity or clinical safety claim.'};save_json(out/'result.json',result);print(json.dumps(result),flush=True)
    if not result['quality_gate']['finite_training']:raise RuntimeError('Nonfinite predictions')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=25);run(p.parse_args())
