"""Low-mass candidate restrictions; synthetic software tests, no camera or training."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import frontend
import planar_force as app
import model
from test_runtime import FakeFrontend,FakeModel,picture

class FixedOutput(FakeModel):
    def __init__(self,value):self.value=value
    def predict(self,uv,valid):return np.array([self.value],np.float32)

class LowMassTests(unittest.TestCase):
    def setUp(self):
        self.scratch=(frontend.ROOT/'_checks/tests').resolve();self.scratch.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.scratch);self.root=Path(self.tmp.name).resolve()
    def tearDown(self):
        if not self.root.is_relative_to(self.scratch):raise RuntimeError('Unsafe cleanup')
        self.tmp.cleanup()
    def make(self,value,sensor=model.CAPTURE_SENSOR):
        s=app.Session(self.root/'session','saved_image_replay',sensor,FixedOutput(value),FakeFrontend())
        p,h=s.save_raw(picture(0),'ref.png');s.set_reference(picture(0),p,h)
        return s
    def close(self,s):
        with contextlib.redirect_stdout(io.StringIO()):s.finish('COMPLETE_REPLAY')
    def test_both_declared_ids_preserve_unresolved_identity(self):
        for sensor in (model.CAPTURE_SENSOR,model.INHERITED_SENSOR):app.declare(sensor,True)
        with self.assertRaises(ValueError):app.declare('HSMVBTS_G01_H20_000099',True)
        s=self.make(-.1);row=s.base_row(0,1,1,0);s.process(row,picture(10));s.write(row);self.close(s)
        meta=json.loads((self.root/'session/session.json').read_text());receipt=json.loads((self.root/'session/receipt.json').read_text())
        self.assertEqual(meta['sensor_id'],model.CAPTURE_SENSOR);self.assertEqual(row['identity_status'],'UNRESOLVED');self.assertEqual(receipt['identity_status'],'UNRESOLVED')
        self.assertNotEqual(meta['capture_snapshot_sensor_id'],meta['inherited_profile_sensor_id'])
    def test_negative_outside_warns_without_clamping(self):
        s=self.make(-2.);row=s.base_row(0,1,1,0);s.process(row,picture(10));s.write(row);self.close(s)
        self.assertEqual(row['fz_raw_n'],-2.);self.assertEqual(row['display_fz_n'],-2.);self.assertEqual(row['valid'],1)
        self.assertEqual(row['status'],app.DEMO_RANGE_WARNING);self.assertEqual(row['quality_status'],'VALID_RAW_FZ')
        self.assertEqual(json.loads((self.root/'session/receipt.json').read_text())['output_range_warning_rows'],1)
    def test_positive_outside_warns_after_display_tare(self):
        s=self.make(.3);first=s.base_row(0,1,1,0);s.process(first,picture(10));s.write(first);s.request_tare(first)
        second=s.base_row(1,2,2,0);s.process(second,picture(10));s.write(second);self.close(s)
        self.assertAlmostEqual(second['fz_raw_n'],.3,places=6);self.assertEqual(second['display_fz_n'],0.)
        self.assertEqual(second['demo_output_range_warning'],app.DEMO_RANGE_WARNING)
    def test_inside_range_does_not_claim_in_distribution(self):
        s=self.make(-.1);row=s.base_row(0,1,1,0);s.process(row,picture(10));s.write(row);self.close(s)
        self.assertEqual(row['demo_output_range_warning'],'');self.assertEqual(row['valid'],1)
        receipt=json.loads((self.root/'session/receipt.json').read_text());self.assertEqual(receipt['readiness'],'NOT_READY')
        self.assertEqual(receipt['range_check_meaning'],'OUTPUT_WARNING_ONLY_NOT_OOD_OR_CONFIDENCE')
    def test_frontend_is_byte_identical_to_verified_runtime(self):
        self.assertEqual(frontend.sha_file(frontend.ROOT/'frontend.py'),'ae4ec141baa8e25ea82cb10e54cdd6da60fb620db2c02a852b8db3641f05fd45')
        self.assertEqual(frontend.sha_file(frontend.ROOT/'frontend_contract.json'),'d40afc9b243d1ae71ce4f27c10a564fb050b8746a10e135b1e5a639001f6ff4e')
        f=frontend.PairFrontend();f.assert_unchanged()
    def test_old_formal_schema_rejected_before_onnx_load(self):
        (self.root/'manifest.json').write_text(json.dumps({'schema':'planar_raw_uv_fz_onnx_v1'}))
        with self.assertRaisesRegex(ValueError,'low-mass adaptation'):model.ForceModel(model_dir=self.root)
    def test_default_is_preregistered_head(self):
        self.assertEqual(model.MODELS,('lowmass_head60','lowmass_aug60'))
    def manifest_fixture(self):
        digest=frontend.sha_file(frontend.ROOT/'frontend_contract.json')
        entry=dict(name='lowmass_head60',qualification='NOT_READY',frontend_contract_sha256=digest,
                   parity=dict(status='PASS_ADAPTATION_REPLAY_ONNX_PARITY',rows=1501,failed_rows=0,batch1_and64_pass=True),
                   checkpoint_sha256='1'*64,code_sha256='2'*64,
                   inputs={'raw_uv':{'dtype':'float32','shape':['N',2,69,82],'units':'pixel'},'valid':{'dtype':'bool','shape':['N',1,69,82]}},
                   output={'dtype':'float32','shape':['N'],'units':'N','axis':'signed_source_Fz'},
                   normalization={'embedded_in_graph':True,'scope':'FROZEN_PRETRAINING_TRAIN_ONLY'})
        return dict(schema='planar_lowmass_demo_candidate_v1',default_model='lowmass_head60',qualification='NOT_READY',
                    calibration_scope='CURRENT_FAMA_SENSOR_0_100G',validation_status='RESUBSTITUTION_FIT_AND_SHARED_REFERENCE_CYCLE_DIAGNOSTICS_ONLY',
                    capture_snapshot_sensor_id=model.CAPTURE_SENSOR,inherited_profile_sensor_id=model.INHERITED_SENSOR,identity_status='UNRESOLVED',
                    frontend_contract_sha256=digest,models=[entry,{**entry,'name':'lowmass_aug60'}])
    def reject(self,value,reason):
        (self.root/'manifest.json').write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,reason):model.ForceModel(model_dir=self.root)
    def test_formal_parity_cannot_qualify_new_candidate(self):
        value=self.manifest_fixture();value['models'][0]['parity']={'status':'PASS_ALL_RECORDED_FRAME_ONNX_PARITY','rows':42465,'failed_rows':0,'batch1_and64_pass':True}
        self.reject(value,'1501-row adaptation')
    def test_incomplete_adaptation_parity_rejected(self):
        value=self.manifest_fixture();value['models'][0]['parity']['failed_rows']=1;self.reject(value,'1501-row adaptation')
    def test_resolved_identity_or_independent_validation_claim_rejected(self):
        for key,val in [('identity_status','VERIFIED'),('validation_status','INDEPENDENT_PASS'),('default_model','lowmass_aug60')]:
            value=self.manifest_fixture();value[key]=val;self.reject(value,'scope/identity/default')
    def test_unknown_extra_candidate_rejected(self):
        value=self.manifest_fixture();value['models'].append({**value['models'][0],'name':'extra'});self.reject(value,'Exactly the two')
    def test_changed_normalization_scope_rejected(self):
        value=self.manifest_fixture();value['models'][0]['normalization']['scope']='TRAIN_ONLY';self.reject(value,'frozen pretraining')

if __name__=='__main__':unittest.main()
