import csv,json,os,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'seed17'))
import lab
from engine import Engine,PACKAGE,digest
from sync_acquisition import FramePacket,ForceSample,ForceMatch

class KnownPrediction:
    binding={'model':'fake fixture','active_profile_sha256':digest(PACKAGE/'HSMVBTS_G01_H20_000011.txt')}
    def reference(self,image):self.ref=image.copy()
    def predict(self,image):
        if image[0,0,0]==255:raise RuntimeError('injected invalid prediction')
        return 0.55

class LabTests(unittest.TestCase):
    def test_weight_binding(self):
        manifest=json.loads((PACKAGE/'manifest.json').read_text())
        self.assertEqual(digest(PACKAGE/'seed17.json'),manifest['weights_sha256'])
        self.assertEqual(digest(PACKAGE/'native/seed17_tracker.dll'),manifest['tracker_sha256'])
        self.assertEqual(digest(PACKAGE/'HSMVBTS_G01_H20_000011.txt'),manifest['active_profile_sha256'])

    def test_synchronized_capture_and_plateau_report(self):
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(camera=0,ft_dll=Path('unused'),profile=PACKAGE/'HSMVBTS_G01_H20_000011.txt',specimen='fake',position='synthetic',output=Path(temp),port='',baud=115200)
            def simulate(_,*,recorder_factory,lab_view):
                r=recorder_factory(args.output,profile_path=args.profile,metadata={'source':'fake'})
                base=time.perf_counter_ns();img=np.zeros((480,640,3),np.uint8)
                def match(ts):
                    vals=(0,0,-1.03,0,0,0)
                    def sample(i):return ForceSample(i,ts+i,ts+i,ts+i,0,i,0,0,'',vals,vals,1,False,True,'OK')
                    return ForceMatch('OK',ts,1,sample(1),sample(2),.5,vals,0)
                packet=lambda i,ts:FramePacket(i,img,ts-1,ts,ts+1,ts+2)
                ref=r.save_reference(packet(1,base),match(base))
                t=time.perf_counter_ns();lab_view.mark(r,1,t)
                r.record_pair(packet(2,t+2),match(t+2),ref)
                r.record_pair(packet(3,t+3),ForceMatch('missing_right',t+3,1),ref)
                img[0,0,0]=255
                r.record_pair(packet(4,t+4),match(t+4),ref)
                r.close();return 0
            with patch('collect_sync_dataset.run_collector',side_effect=simulate):
                self.assertEqual(lab.capture(args,KnownPrediction()),0)
            report=next(Path(temp).glob('*report_*'))
            def rows(name):
                with (report/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
            frames=rows('frames.csv');self.assertEqual(len(frames),3)
            self.assertAlmostEqual(float(frames[0]['error_N']),-.48)
            self.assertEqual(frames[1]['meter_press_N'],'')
            self.assertEqual(frames[2]['press_N'],'')
            stats=rows('plateaus.csv')[0];self.assertEqual(stats['valid_pairs'],'1');self.assertEqual(stats['frames'],'3')
            self.assertAlmostEqual(float(stats['mae_N']),.48)

    @unittest.skipUnless(os.environ.get('HSMVBTS_OPENCV_DLL'),'requires device Windows SDK runtime')
    def test_native_generated_fixture(self):
        e=Engine(os.environ['HSMVBTS_OPENCV_DLL'])
        try:
            ref,img=lab.synthetic()
            with self.assertRaises(RuntimeError):e.predict(img)
            e.reference(ref);gold=json.loads((PACKAGE/'selftest_expected.json').read_text())
            for _ in range(4):self.assertAlmostEqual(e.predict(img),gold['generated_image_N'],places=6)
            with self.assertRaises(ValueError):e.predict(img[:200])
        finally:e.close()
