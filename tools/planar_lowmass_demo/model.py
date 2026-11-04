"""CPU ONNX Runtime only. Inputs and physical Fz output are strictly checked."""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from frontend import ROOT, sha_bytes, sha_file

MODELS=('lowmass_head60','lowmass_aug60')
CAPTURE_SENSOR='HSMVBTS_G01_H20_000015'
INHERITED_SENSOR='HSMVBTS_G01_H20_000011'

class ForceModel:
    def __init__(self,name=None,model_dir=None):
        folder=Path(model_dir or ROOT/'models').resolve()
        manifest_path=folder/'manifest.json';raw=manifest_path.read_bytes();manifest=json.loads(raw.decode('utf-8-sig'))
        if manifest.get('schema')!='planar_lowmass_demo_candidate_v1': raise ValueError('Expected low-mass adaptation candidate manifest, not original formal-model schema')
        required=dict(default_model='lowmass_head60',qualification='NOT_READY',calibration_scope='CURRENT_FAMA_SENSOR_0_100G',
                      validation_status='RESUBSTITUTION_FIT_AND_SHARED_REFERENCE_CYCLE_DIAGNOSTICS_ONLY',
                      capture_snapshot_sensor_id=CAPTURE_SENSOR,inherited_profile_sensor_id=INHERITED_SENSOR,identity_status='UNRESOLVED')
        if any(manifest.get(k)!=v for k,v in required.items()):raise ValueError('Calibration-only scope/identity/default contract differs')
        if manifest.get('frontend_contract_sha256')!=sha_file(ROOT/'frontend_contract.json'): raise ValueError('Model/frontend contract mismatch')
        entries=manifest['models'];byname={r['name']:r for r in entries}
        if len(entries)!=2 or set(byname)!=set(MODELS): raise ValueError('Exactly the two low-mass candidates are required')
        self.name=name or manifest['default_model']
        if self.name not in MODELS: raise ValueError('Unknown model; no fallback')
        self.info=byname[self.name]
        if self.info.get('qualification')!='NOT_READY': raise ValueError('Unexpected qualification claim')
        parity=self.info.get('parity',{})
        if parity.get('status')!='PASS_ADAPTATION_REPLAY_ONNX_PARITY' or parity.get('rows')!=1501 or parity.get('failed_rows')!=0 or parity.get('batch1_and64_pass') is not True:
            raise ValueError('Model has no completed 1501-row adaptation-replay ONNX parity acceptance')
        for key in ('checkpoint_sha256','code_sha256'):
            value=self.info.get(key,'')
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing candidate provenance SHA256: '+key)
        if self.info.get('frontend_contract_sha256')!=manifest['frontend_contract_sha256']: raise ValueError('Model entry frontend contract differs')
        if self.info.get('inputs')!={'raw_uv':{'dtype':'float32','shape':['N',2,69,82],'units':'pixel'},'valid':{'dtype':'bool','shape':['N',1,69,82]}}:
            raise ValueError('Manifest input type/shape/units differ')
        if self.info.get('output')!={'dtype':'float32','shape':['N'],'units':'N','axis':'signed_source_Fz'}:
            raise ValueError('Manifest output type/shape/units differ')
        if self.info.get('normalization',{}).get('embedded_in_graph') is not True or self.info.get('normalization',{}).get('scope')!='FROZEN_PRETRAINING_TRAIN_ONLY':
            raise ValueError('Graph must preserve the original frozen pretraining TRAIN normalization')
        path=(folder/self.info['onnx_file']).resolve()
        if not path.is_relative_to(folder) or path.suffix!='.onnx': raise ValueError('ONNX path outside models directory')
        blob=path.read_bytes()
        if sha_bytes(blob)!=self.info['sha256']: raise ValueError('ONNX SHA256 mismatch')
        if self.info['input_names']!={'uv':'raw_uv','valid':'valid'} or self.info['output_name']!='signed_fz_n': raise ValueError('Unsupported graph names')
        self.sources={manifest_path:sha_bytes(raw),path:sha_bytes(blob),ROOT/'frontend_contract.json':manifest['frontend_contract_sha256'],Path(__file__).resolve():sha_file(__file__)}
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(blob,sess_options=options,providers=['CPUExecutionProvider'])
        inputs={v.name:v for v in self.session.get_inputs()};outputs=self.session.get_outputs()
        if set(inputs)!={'raw_uv','valid'} or len(outputs)!=1 or outputs[0].name!='signed_fz_n': raise ValueError('ONNX I/O names differ')
        for name,dtype,shape in [('raw_uv','tensor(float)',[2,69,82]),('valid','tensor(bool)',[1,69,82])]:
            node=inputs[name]
            if node.type!=dtype or len(node.shape)!=4 or node.shape[1:]!=shape or isinstance(node.shape[0],int): raise ValueError('ONNX input dtype/shape differs')
        node=outputs[0]
        if node.type!='tensor(float)' or len(node.shape)!=1 or isinstance(node.shape[0],int): raise ValueError('Expected dynamic [N] physical float32 Fz output')
        self.assert_unchanged()

    def assert_unchanged(self):
        if any(sha_file(p)!=h for p,h in self.sources.items()): raise ValueError('Model/manifest/source changed')

    def predict(self,uv,valid):
        if not isinstance(uv,np.ndarray) or uv.dtype!=np.float32 or uv.ndim!=4 or uv.shape[1:]!=(2,69,82) or uv.shape[0]<1 or not np.isfinite(uv).all(): raise ValueError('Expected finite raw_uv float32 [N,2,69,82]')
        if not isinstance(valid,np.ndarray) or valid.dtype!=np.bool_ or valid.shape!=(uv.shape[0],1,69,82): raise ValueError('Expected valid bool [N,1,69,82]')
        if np.any(valid.mean(axis=(1,2,3))<.2): raise ValueError('Valid grid fraction below 20%; no prediction/hold-last')
        output=self.session.run(['signed_fz_n'],{'raw_uv':uv,'valid':valid})[0]
        if output.dtype!=np.float32 or output.shape!=(uv.shape[0],) or not np.isfinite(output).all(): raise ValueError('Nonfinite or malformed physical Fz output')
        return output
