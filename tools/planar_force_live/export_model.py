"""Export one trusted, hash-bound planar CNN; runtime needs no Torch.

This script contains the frozen inference architecture, not training code.
Original 17x20 adaptive average pooling is expressed as exact floor/ceil regions.
"""
import argparse,hashlib,io,json,math,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
import onnx
import onnxruntime as ort

TRAINING_CODE_SHA='cf07dcbdddce73fb3c1b99db9cfdabcaec3d1ee0c61069fdd17c9d072c71aebb'
INPUT_SHAPE=(2,69,82)
SCHEMA='planar_raw_uv_fz_onnx_v1'

def digest(b):return hashlib.sha256(b).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)

class FrozenCNN(nn.Module):
    def __init__(self,xy):
        super().__init__();self.xy=xy
        self.conv=nn.Sequential(nn.Conv2d(5 if xy else 3,16,3,padding=1),nn.GELU(),nn.AvgPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.GELU(),nn.AvgPool2d(2),
            nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(512,64),nn.GELU(),nn.Linear(64,1))

class ExplicitAdaptivePool(nn.Module):
    def forward(self,x):
        # Exact adaptive pooling bins: floor(i*input/output), ceil((i+1)*input/output).
        # H=17 overlaps at4,8,12, as in original PyTorch AdaptiveAvgPool2d.
        rows=[]
        for y in range(4):
            cols=[]
            for z in range(4):
                patch=x[:,:,math.floor(y*17/4):math.ceil((y+1)*17/4),math.floor(z*20/4):math.ceil((z+1)*20/4)]
                cols.append(patch.mean(dim=(2,3),keepdim=True))
            rows.append(torch.cat(cols,dim=3))
        return torch.cat(rows,dim=2)

class RawUVGraph(nn.Module):
    def __init__(self,ck):
        super().__init__();self.network=FrozenCNN(ck['xy']);self.network.load_state_dict(ck['model'],strict=True)
        self.network.conv[8]=ExplicitAdaptivePool();self.xy=bool(ck['xy'])
        self.register_buffer('input_mean',torch.as_tensor(ck['input_mean'],dtype=torch.float32).reshape(1,2,1,1))
        self.register_buffer('input_scale',torch.as_tensor(ck['input_scale'],dtype=torch.float32).reshape(1,2,1,1))
        self.register_buffer('target_mean',torch.tensor(ck['target_mean'],dtype=torch.float32))
        self.register_buffer('target_scale',torch.tensor(ck['target_scale'],dtype=torch.float32))
        yy,xx=torch.meshgrid(torch.linspace(-1,1,69),torch.linspace(-1,1,82),indexing='ij')
        self.register_buffer('fixed_xy',torch.stack([xx,yy])[None])
    def forward(self,raw_uv,valid):
        x=(raw_uv-self.input_mean)/self.input_scale
        x=torch.where(valid,x,torch.zeros_like(x))
        x=torch.cat([x,valid.to(dtype=torch.float32)],dim=1)
        if self.xy:x=torch.cat([x,self.fixed_xy.expand(raw_uv.shape[0],-1,-1,-1)],dim=1)
        p=self.network.head(self.network.conv(x)).squeeze(1)
        return p*self.target_scale+self.target_mean

def export_one(checkpoint,expected_sha,output,name):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    path=output/(name+'.onnx');meta_path=output/(name+'.export.json')
    if path.exists() or meta_path.exists():raise FileExistsError('Refuse overwriting export '+name)
    blob=Path(checkpoint).read_bytes()
    if digest(blob)!=expected_sha:raise ValueError('Trusted checkpoint SHA mismatch')
    loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);cfg=loaded['config'];ck=loaded['inference']
    if cfg.get('code_sha256')!=TRAINING_CODE_SHA or cfg.get('synthetic') is not False or cfg.get('epochs')!=8 or cfg.get('supervised_axes')!=['Fz'] or tuple(cfg.get('input_shape',[]))!=INPUT_SHAPE:raise ValueError('Unexpected frozen formal Fz architecture/contract')
    if ck.get('kind') not in ('cnn_uv','cnn_xy') or ck['kind']!=cfg['kind'] or bool(ck['xy'])!=(ck['kind']=='cnn_xy'):raise ValueError('Wrong CNN variant')
    for key in ('input_mean','input_scale'):
        if np.asarray(ck[key]).shape!=(2,) or not np.isfinite(ck[key]).all():raise ValueError('Invalid input normalization')
    if np.any(ck['input_scale']<=0) or not math.isfinite(ck['target_mean']) or not math.isfinite(ck['target_scale']) or ck['target_scale']<=0:raise ValueError('Invalid scale')
    torch.set_num_threads(2);graph=RawUVGraph(ck).eval()
    raw=torch.zeros((1,*INPUT_SHAPE),dtype=torch.float32);valid=torch.ones((1,1,69,82),dtype=torch.bool)
    torch.onnx.export(graph,(raw,valid),str(path),opset_version=17,dynamo=False,do_constant_folding=True,
        input_names=['raw_uv','valid'],output_names=['signed_fz_n'],
        dynamic_axes={'raw_uv':{0:'N'},'valid':{0:'N'},'signed_fz_n':{0:'N'}})
    model=onnx.load(str(path));onnx.checker.check_model(model,full_check=True)
    if any(i.data_location==onnx.TensorProto.EXTERNAL for i in model.graph.initializer):raise ValueError('Graph must be one self-contained file')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(path.read_bytes(),sess_options=options,providers=['CPUExecutionProvider'])
    for batch in (1,64):
        result=session.run(['signed_fz_n'],{'raw_uv':np.zeros((batch,*INPUT_SHAPE),np.float32),'valid':np.ones((batch,1,69,82),bool)})[0]
        if result.shape!=(batch,) or result.dtype!=np.float32 or not np.isfinite(result).all():raise ValueError('Dynamic batch smoke failed')
    meta={'schema':SCHEMA,'name':name,'kind':ck['kind'],'seed':cfg['seed'],'onnx_file':path.name,'sha256':digest(path.read_bytes()),
        'input_names':{'uv':'raw_uv','valid':'valid'},'output_name':'signed_fz_n',
        'inputs':{'raw_uv':{'dtype':'float32','shape':['N',2,69,82],'units':'pixel'},'valid':{'dtype':'bool','shape':['N',1,69,82]}},
        'output':{'dtype':'float32','shape':['N'],'units':'N','axis':'signed_source_Fz'},
        'normalization':{'scope':'TRAIN_ONLY','input_mean':np.asarray(ck['input_mean']).tolist(),'input_scale':np.asarray(ck['input_scale']).tolist(),
            'target_mean':ck['target_mean'],'target_scale':ck['target_scale'],'arithmetic':'float32','embedded_in_graph':True},
        'xy_channels_embedded':bool(ck['xy']),'xy_definition':'image-fixed x,y linspace[-1,1], not stage coordinates',
        'checkpoint_sha256':expected_sha,'training_code_sha256':TRAINING_CODE_SHA,'export_code_sha256':digest(Path(__file__).read_bytes()),
        'invalid_cell_policy':'where(valid,normalized_uv,0), append float32 validity channel',
        'frame_quality_gate_required':'Reject nonfinite raw_uv or per-frame valid fraction <0.2 before graph invocation',
        'adaptive_pool_export':'Exact floor/ceil overlapping region means for17x20->4x4; no training changes',
        'onnx_opset':17,'onnx_checker':'PASS','dynamic_batches_checked':[1,64],
        'output_policy':{'sign_fit':False,'tare':False,'smooth':False,'clip':False,'hold_last':False},
        'qualification':'NOT_READY','physical_accuracy':'UNVERIFIED','sdk_drop_in_compatible':False,
        'parity':{'status':'NOT_YET_EVALUATED'},'versions':{'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,'numpy':np.__version__}}
    save(meta_path,meta)
    return meta

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--expected-checkpoint-sha256',required=True);p.add_argument('--output',required=True);p.add_argument('--name',required=True);a=p.parse_args()
    if Path(a.name).name!=a.name:raise ValueError('Model name must be a single filename stem')
    print(json.dumps(export_one(a.checkpoint,a.expected_checkpoint_sha256,a.output,a.name)))
if __name__=='__main__':main()
