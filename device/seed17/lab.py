"""Read-only H20 ball-probe trial: explicit camera and FT capture, or offline inference."""
import argparse,bisect,csv,json,os,sys,time
from pathlib import Path
import numpy as np
from engine import Engine,PACKAGE,digest
sys.path.insert(0,str(PACKAGE.parent))

def synthetic():
    # Reproducible generated texture, unrelated to customer images.
    y,x=np.indices((480,640),dtype='int64')
    h=(x*73856093)^(y*19349663);blue=((h^(h>>9)^(h>>17))%256).astype(np.uint8)
    ref=np.stack((blue,blue,blue),axis=2)
    cur=ref.copy();cur[180:270,260:350]=ref[180:270,258:348]
    return ref,cur

class View:
    def __init__(self):
        self.markers=[];self.latest=None
    def mark(self,recorder,stage,when):
        self.markers.append((when,stage,len(self.markers)+1))
        recorder.event('seed17_plateau_marker',stage=stage,marker_time_ns=when,marker_index=len(self.markers))
    def at(self,ts):
        i=bisect.bisect_right([t[0] for t in self.markers],ts)-1
        return self.markers[i][1:] if i>=0 else (None,0)
    def key(self,key,controller,recorder):
        if key in (ord('r'),ord('R')):
            self.mark(recorder,None,time.perf_counter_ns())
            self.latest=None
        if controller.recording and key in [ord(x) for x in '0135bB']:
            self.mark(recorder,None if chr(key).lower()=='b' else int(chr(key)),time.perf_counter_ns())
    def draw(self,img,packet,controller):
        import cv2
        current=self.latest
        label='Record with R; mark stable interval: 0/1/3/5, B before changing load'
        if controller.recording and current and current['ref_id']==controller.reference_id:
            ref='invalid' if current['ref'] is None else f"{current['ref']:.3f}"
            val='invalid' if current['pred'] is None else f"{current['pred']:.3f}"
            age=max(0,(time.perf_counter_ns()-current['ts'])/1e6)
            label=f"Pair #{current['seq']} age {age:.0f}ms | Fz press N: model {val}   FT {ref}"
        cv2.rectangle(img,(0,83),(img.shape[1],140),(20,20,20),-1)
        cv2.putText(img,label,(10,106),cv2.FONT_HERSHEY_SIMPLEX,.47,(80,245,250),1)
        stage,_=self.at(packet.grab_end_ns)
        cv2.putText(img,f"Plateau {stage if stage is not None else '-'} | 0/1/3/5 = start, B = end",(10,130),cv2.FONT_HERSHEY_SIMPLEX,.47,(220,220,220),1)

def capture(args,engine):
    from sync_acquisition import SessionRecorder
    from collect_sync_dataset import run_collector
    view=View()
    capture_dir=[]
    class LabRecorder(SessionRecorder):
        def __init__(self,*a,**kw):
            kw['metadata']={**kw['metadata'],'seed17':engine.binding,'specimen':args.specimen,'position':args.position}
            super().__init__(*a,**kw);self.model_ref=None
            capture_dir.append(self.session_dir)
        def save_reference(self,packet,match,reason='user'):
            rid=super().save_reference(packet,match,reason)
            engine.reference(packet.image);self.model_ref=rid
            view.mark(self,None,time.perf_counter_ns())
            return rid
        def record_pair(self,packet,match,reference_id=None,prediction=None,extra=None):
            start=time.perf_counter_ns();value=None;error=None
            try:
                if reference_id!=self.model_ref:raise ValueError('reference mismatch')
                value=engine.predict(packet.image)
            except (OSError,RuntimeError,ValueError) as exc:error=str(exc)
            stage,index=view.at(packet.grab_end_ns)
            pred={'force':[None,None,-value if value is not None else None,None,None,None],
                  'press_N':value,'status':'OK' if error is None else 'INVALID','error':error,
                  'model':'uv_mlp_17','inference_ns':time.perf_counter_ns()-start}
            row=super().record_pair(packet,match,reference_id,pred,{**(extra or {}),'plateau_target_N':stage,'plateau_id':index})
            view.latest={'seq':packet.seq,'ts':packet.grab_end_ns,'ref_id':reference_id,
                         'ref':-match.label[2] if match.valid else None,'pred':value}
            return row
    if digest(args.profile)!=engine.binding['active_profile_sha256']:
        raise ValueError('profile differs from the verified HSMVBTS_G01_H20_000011 TXT')
    status=run_collector(['--camera-index',str(args.camera),'--ft-dll',str(args.ft_dll),'--profile',str(args.profile),
                         '--output-dir',str(args.output),'--port',args.port,'--baud-rate',str(args.baud)],
                         recorder_factory=LabRecorder,lab_view=view)
    if status==0 and capture_dir:
        with (capture_dir[0]/'capture_sync.csv').open(encoding='utf-8-sig',newline='') as f:
            saved=next(csv.DictReader(f),None)
        if saved:
            from evaluate_lab import evaluate
            print('Report:',evaluate(capture_dir[0]))
    return status

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for command in ['selftest','predict','capture']:
        q=sub.add_parser(command)
        q.add_argument('--opencv-dll',type=Path,default=os.environ.get('HSMVBTS_OPENCV_DLL'),
                       required=not bool(os.environ.get('HSMVBTS_OPENCV_DLL')),help='matching SDK opencv_world440.dll')
        if command=='predict':
            q.add_argument('--reference',type=Path,required=True);q.add_argument('--image',type=Path,required=True)
        if command=='capture':
            q.add_argument('--camera',type=int,required=True);q.add_argument('--ft-dll',type=Path,required=True)
            q.add_argument('--profile',type=Path,required=True);q.add_argument('--specimen',required=True);q.add_argument('--position',required=True)
            q.add_argument('--port',default='');q.add_argument('--baud',default=115200,type=int)
            q.add_argument('--output',type=Path,default=PACKAGE.parents[1]/'local_data/seed17')
    q=sub.add_parser('report');q.add_argument('session',type=Path)
    args=p.parse_args(argv)
    if args.action=='report':
        from evaluate_lab import evaluate
        print(evaluate(args.session));return 0
    e=Engine(args.opencv_dll)
    try:
        if args.action=='selftest':
            ref,img=synthetic();e.reference(ref);zero=e.predict(ref);v=[e.predict(img) for _ in range(8)]
            golden=json.loads((PACKAGE/'selftest_expected.json').read_text())
            if abs(zero-golden['reference_N'])>1e-6 or any(abs(y-golden['generated_image_N'])>1e-6 for y in v) or np.ptp(v)>1e-8:
                raise ValueError(f'selftest mismatch: {zero}, {v}')
            print(json.dumps({'selftest':'PASS','reference_N':zero,'synthetic_N':v[0],'repeat_span_N':float(np.ptp(v))}));return 0
        if args.action=='predict':
            import cv2
            read=lambda path:cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            e.reference(read(args.reference));v=e.predict(read(args.image))
            print(json.dumps({'press_N':v,'signed_Fz_N':-v,'status':'OK'}));return 0
        return capture(args,e)
    finally:e.close()
if __name__=='__main__':raise SystemExit(main())
