"""Software-only unittest suite. No test opens a camera or trains a model."""
import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import frontend
import planar_force as app
from model import ForceModel

def picture(level):return np.full((480,640,3),level,np.uint8)
class FakeFrontend:
    sources={}
    def process(self,reference,current):
        valid=np.ones((1,1,69,82),bool)
        if current[0,0,0]==99:valid[:]=False
        return dict(uv=np.full((1,2,69,82),-float(current[0,0,0])/100,np.float32),valid=valid,
                    quality_valid_fraction=float(valid.mean()),confidence=valid[0,0].astype(np.float32))
class FakeModel:
    name='SYNTHETIC_ONLY';sources={};info={}
    def predict(self,uv,valid):return uv[:,0,0,0].copy()
class FakeCapture:
    def __init__(self,frames):self.frames=iter(frames);self.released=False
    def isOpened(self):return True
    def read(self):return next(self.frames,(False,None))
    def release(self):self.released=True

class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.scratch=(frontend.ROOT/'_checks/tests').resolve();self.scratch.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.scratch);self.root=Path(self.tmp.name).resolve()
    def tearDown(self):
        if not self.root.is_relative_to(self.scratch):raise RuntimeError('Refuse cleanup outside test scratch')
        self.tmp.cleanup()
    def args(self,output):return argparse.Namespace(sensor_id=frontend.SENSOR_ID,declare_unloaded=True,model=None,output=str(output),normal_sign=None,
          camera_index=3,backend='any',duration_seconds=30,max_frames=7,no_window=True)
    def simulate(self,frames,keys=None,sign=None):
        cap=FakeCapture(frames);out=self.root/'session';args=self.args(out);args.max_frames=len(frames);args.normal_sign=sign
        keys=iter(keys or [-1]*len(frames))
        with contextlib.redirect_stdout(io.StringIO()):code=app.live(args,FakeModel(),FakeFrontend(),lambda i,b:cap,lambda im,row,s:next(keys,-1))
        rows=list(csv.DictReader(io.StringIO((out/'predictions.csv').read_text(encoding='utf-8'))))
        return code,rows,json.loads((out/'receipt.json').read_text()),cap
    def test_step_ramp_invalid_no_hold(self):
        code,rows,r,cap=self.simulate([(True,picture(x)) for x in [0,1,10,20,99,21,22]])
        self.assertEqual(code,2);self.assertEqual(len(rows),7);self.assertEqual(rows[4]['fz_raw_n'],'');self.assertEqual(rows[4]['display_fz_n'],'')
        for i,value in [(1,-.01),(2,-.1),(3,-.2),(5,-.21),(6,-.22)]:self.assertAlmostEqual(float(rows[i]['fz_raw_n']),value,places=6)
        self.assertEqual(r['invalid_rows'],1);self.assertFalse(r['camera_accessed']);self.assertEqual(r['capture_mode'],'SIMULATED_TEST');self.assertTrue(cap.released)
    def test_once_tare_preserves_raw_and_logged_transition(self):
        code,rows,r,cap=self.simulate([(True,picture(x)) for x in [0,10,20,30]],[-1,ord('t'),ord('t'),-1],-1)
        self.assertEqual(code,0);self.assertAlmostEqual(float(rows[2]['fz_raw_n']),-.2,places=6);self.assertAlmostEqual(float(rows[2]['display_fz_n']),.1,places=6)
        events=[json.loads(s) for s in (self.root/'session/events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind']=='USER_ONE_TIME_TARE' for e in events),1);self.assertEqual(sum(e['kind']=='TARE_REJECTED_ALREADY_SET' for e in events),1)
    def test_read_failures_retained_and_abort(self):
        code,rows,r,cap=self.simulate([(True,picture(0))]+[(False,None)]*5)
        self.assertEqual(code,2);self.assertEqual(len(rows),6);self.assertEqual(r['status'],'FAIL_CONSECUTIVE_INVALID');self.assertTrue(all(x['fz_raw_n']=='' for x in rows))
    def test_wrong_geometry_retained_raw(self):
        code,rows,r,cap=self.simulate([(True,picture(0)),(True,np.zeros((100,100,3),np.uint8))])
        self.assertEqual(code,2);self.assertTrue(Path(rows[1]['raw_image_path']).is_file());self.assertEqual(rows[1]['valid'],'0')
    def test_raw_save_before_overlay(self):
        cap=FakeCapture([(True,picture(0)),(True,picture(10))]);args=self.args(self.root/'rawfirst');args.max_frames=2
        def mutate(im,row,session):im[:]=255;return -1
        with contextlib.redirect_stdout(io.StringIO()):app.live(args,FakeModel(),FakeFrontend(),lambda i,b:cap,mutate)
        loaded=cv2.imread(str(self.root/'rawfirst/raw/frame_000001.png'))
        self.assertTrue(np.all(loaded==10))
    def test_fresh_output_refuses_overwrite(self):
        self.root.joinpath('exists').mkdir()
        with self.assertRaises(FileExistsError):app.Session(self.root/'exists','saved_image_replay',frontend.SENSOR_ID,FakeModel(),FakeFrontend())
    def test_identity_and_unloaded_are_required(self):
        with self.assertRaises(ValueError):app.declare('unknown',True)
        with self.assertRaises(ValueError):app.declare(frontend.SENSOR_ID,False)
    def test_reference_cannot_update(self):
        s=app.Session(self.root/'fixed','saved_image_replay',frontend.SENSOR_ID,FakeModel(),FakeFrontend());path,h=s.save_raw(picture(0),'ref.png');s.set_reference(picture(0),path,h)
        with self.assertRaises(ValueError):s.set_reference(picture(1),path,h)
        with contextlib.redirect_stdout(io.StringIO()):s.finish('COMPLETE_REPLAY')
    def test_plot_breaks_invalid_retains_microstep(self):
        rows=[dict(timestamp_ns=i*1_000_000_000,valid=v,fz_raw_n=x) for i,(x,v) in enumerate([(-.01,1),(-.010001,1),('',0),(-.2,1)])]
        segments=app.history_segments(rows,'fz_raw_n',3_000_000_000)
        self.assertEqual(len(segments),2);self.assertEqual([v for s in segments for _,v in s],[-.01,-.010001,-.2])
    def test_saved_missing_frame_retained(self):
        ref=self.root/'ref.png';im=self.root/'图像 with spaces.png';cv2.imwrite(str(ref),picture(0));cv2.imwrite(str(im),picture(10))
        args=self.args(self.root/'saved');args.reference=str(ref);args.images=[str(im),str(self.root/'missing.png'),str(im)]
        with contextlib.redirect_stdout(io.StringIO()):code=app.replay(args,FakeModel(),FakeFrontend())
        rows=list(csv.DictReader(io.StringIO((self.root/'saved/predictions.csv').read_text(encoding='utf-8'))));self.assertEqual(code,2);self.assertEqual(len(rows),3);self.assertEqual(rows[1]['valid'],'0')
    def test_saved_bad_reference_retains_all_requested_frames(self):
        im=self.root/'im.png';cv2.imwrite(str(im),picture(10));args=self.args(self.root/'badref');args.reference=str(self.root/'missing.png');args.images=[str(im),str(im)]
        with contextlib.redirect_stdout(io.StringIO()):code=app.replay(args,FakeModel(),FakeFrontend())
        rows=list(csv.DictReader(io.StringIO((self.root/'badref/predictions.csv').read_text(encoding='utf-8'))));self.assertEqual(code,2);self.assertEqual(len(rows),2);self.assertTrue(all('REFERENCE_ERROR' in r['error'] for r in rows))
    def test_strict_pixel_shape(self):
        for a in [np.zeros((480,640),np.uint8),np.zeros((480,640,3),np.float32),np.zeros((480,639,3),np.uint8)]:
            with self.assertRaises(ValueError):frontend.checked_image(a)
    def test_portable_dis_no_native_or_torch(self):
        f=frontend.PairFrontend();image=np.random.default_rng(17).integers(20,230,(480,640,3),dtype=np.uint8)
        with mock.patch.object(f.old.base.legacy,'ensure_native',side_effect=AssertionError('Native compiler must not run')):result=f.process(image,image)
        self.assertEqual(result['uv'].shape,(1,2,69,82));self.assertGreater(result['quality_valid_fraction'],.2);self.assertNotIn('torch',sys.modules)
    def test_inference_input_gate_without_model(self):
        model=ForceModel.__new__(ForceModel)
        uv=np.zeros((1,2,69,82),np.float32);valid=np.ones((1,1,69,82),bool)
        for a,b in [(uv.astype(np.float64),valid),(uv,valid.astype(np.float32)),(uv[:,:,:68],valid),(np.full_like(uv,np.nan),valid),(uv,np.zeros_like(valid))]:
            with self.assertRaises(ValueError):model.predict(a,b)

if __name__=='__main__':unittest.main()
