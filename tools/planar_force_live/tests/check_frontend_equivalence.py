"""Optional private-data audit; outputs aggregate hashes/results, never copies images."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frontend import PairFrontend,read_image,sha_file
from model import ForceModel,MODELS

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-module',required=True);p.add_argument('--pairs-manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    spec=importlib.util.spec_from_file_location('comparison_baseline',a.baseline_module);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    baseline=old.PairFrontend('dis_pairwise');portable=PairFrontend()
    manifest=json.loads(Path(a.pairs_manifest).read_text(encoding='utf-8-sig'));refs={r['reference_id']:r for r in manifest['references']}
    models=[ForceModel(n) for n in MODELS];maximums={k:0. for k in ['uv','confidence','fz']};valid_diff=reason_diff=0
    for row in manifest['frames']:
        ref=refs[row['reference_id']];reference,_=read_image(ref['reference_path'],ref['reference_sha256']);current,_=read_image(row['current_path'],row['current_sha256'])
        first=baseline.process(reference,current);second=portable.process(reference,current)
        uv=np.transpose(first['uv'],(2,0,1))[None];valid=first['valid'][None,None]
        maximums['uv']=max(maximums['uv'],float(np.max(abs(uv-second['uv']))));maximums['confidence']=max(maximums['confidence'],float(np.max(abs(first['confidence']-second['confidence']))))
        valid_diff+=int(np.count_nonzero(valid!=second['valid']));reason_diff+=int(np.count_nonzero(first['reason_bits']!=second['reason_bits']))
        for model in models:maximums['fz']=max(maximums['fz'],float(np.max(abs(model.predict(uv,valid)-model.predict(second['uv'],second['valid'])))))
    portable.assert_unchanged()
    for m in models:m.assert_unchanged()
    passed=all(v==0 for v in maximums.values()) and valid_diff==reason_diff==0
    receipt=dict(status='PASS_EXACT_FLOAT_FRONTEND_EQUIVALENCE' if passed else 'FAIL',readiness='NOT_READY',real_image_pairs=len(manifest['frames']),
        onnx_models_compared=len(models),max_absolute_difference=maximums,valid_disagreements=valid_diff,reason_disagreements=reason_diff,
        baseline_module_sha256=sha_file(a.baseline_module),pair_manifest_sha256=sha_file(a.pairs_manifest),
        frontend_contract_sha256=sha_file(Path(__file__).resolve().parents[1]/'frontend_contract.json'),camera_accessed=False,images_copied_or_uploaded=False)
    out.write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt));return 0 if passed else 2
if __name__=='__main__':raise SystemExit(main())
