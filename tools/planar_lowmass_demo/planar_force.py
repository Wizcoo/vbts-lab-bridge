"""Explicit user-operated live acquisition or saved-image Fz replay; no Torch."""
import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from frontend import ROOT, SENSOR_ID, PairFrontend, checked_image, read_image, sha_bytes, sha_file
from model import ForceModel, MODELS, CAPTURE_SENSOR, INHERITED_SENSOR

ALLOWED_DECLARED_SENSOR_IDS=(CAPTURE_SENSOR,INHERITED_SENSOR)
DEMO_OUTPUT_RANGE=(-1.15,.15)
DEMO_RANGE_WARNING='OUTSIDE_EXPECTED_DEMO_OUTPUT_RANGE'

FIELDS=['attempt_index','frame_id','timestamp_ns','host_wall_time_ns','timestamp_kind','sensor_id','model',
        'fz_raw_n','valid','quality_valid_fraction','confidence_mean','quality_status','error','status','display_fz_n',
        'demo_output_range_warning','identity_status','capture_snapshot_sensor_id','inherited_profile_sensor_id','calibration_scope',
        'normal_sign','tare_offset_raw_n','tare_event_timestamp_ns','reference_path','reference_sha256',
        'raw_image_path','raw_image_sha256','capture_read_ms','frontend_ms','model_ms','raw_save_ms','processing_ms']

def save_json(path,value): Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def declare(sensor,unloaded):
    if sensor not in ALLOWED_DECLARED_SENSOR_IDS: raise ValueError('Explicit sensor-id must be FaMa snapshot '+CAPTURE_SENSOR+' or user-declared inherited '+INHERITED_SENSOR+'; actual hardware identity remains UNRESOLVED')
    if unloaded is not True: raise ValueError('Explicit --declare-unloaded is required for the fixed reference')

class Session:
    def __init__(self,output,mode,sensor,model,frontend,normal_sign=None,capture_mode='SAVED_FILES'):
        self.out=Path(output).resolve();self.out.mkdir(parents=True,exist_ok=False)
        (self.out/'raw').mkdir()
        self.file=(self.out/'predictions.csv').open('x',encoding='utf-8',newline='')
        self.writer=csv.DictWriter(self.file,fieldnames=FIELDS);self.writer.writeheader();self.file.flush()
        self.events=(self.out/'events.jsonl').open('x',encoding='utf-8')
        self.mode=mode;self.sensor=sensor;self.model=model;self.frontend=frontend;self.sign=normal_sign;self.capture_mode=capture_mode
        self.backend_opened=False
        self.capture_provider={'SAVED_FILES':'saved_images','SIMULATED_TEST':'injected_test_stream','USER_OPENCV_CAPTURE':'opencv'}[capture_mode]
        self.tare=None;self.tare_time=None;self.rows=0;self.invalid=0;self.range_warnings=0;self.reference=None;self.refpath='';self.refhash=''
        self.history=[]
        self.sources={**getattr(model,'sources',{}),**getattr(frontend,'sources',{}),Path(__file__).resolve():sha_file(__file__)}
        save_json(self.out/'session.json',dict(schema='planar_force_recording_v1',mode=mode,capture_mode=capture_mode,capture_provider=self.capture_provider,
            camera_requested=capture_mode=='USER_OPENCV_CAPTURE',sensor_id=sensor,
            identity_evidence='DECLARED_ID_WITH_UNRESOLVED_CAPTURE_VS_INHERITED_PROFILE_MISMATCH',identity_status='UNRESOLVED',
            capture_snapshot_sensor_id=CAPTURE_SENSOR,inherited_profile_sensor_id=INHERITED_SENSOR,
            calibration_scope='CURRENT_FAMA_SENSOR_0_100G',validation_status='RESUBSTITUTION_FIT_AND_SHARED_REFERENCE_CYCLE_DIAGNOSTICS_ONLY',
            expected_demo_output_range_n=list(DEMO_OUTPUT_RANGE),range_check_meaning='OUTPUT_WARNING_ONLY_NOT_OOD_OR_CONFIDENCE',
            contact_position_invariance_verified=False,cross_sensor_generalization_verified=False,reference_unloaded='EXPLICIT_USER_DECLARATION',
            model=model.name,model_manifest=getattr(model,'info',{}),normal_sign=normal_sign,
            timestamp_kind='host_receive_monotonic_ns_NOT_camera_exposure',readiness='NOT_READY',
            hardware_tested_by_agent=False,original_signed_fz_preserved=True,automatic_tare=False,smoothing=False,hold_last=False,
            source_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:h for p,h in self.sources.items()}))
    def event(self,kind,**details):
        self.events.write(json.dumps(dict(kind=kind,event_host_monotonic_ns=time.perf_counter_ns(),**details),ensure_ascii=False,allow_nan=False)+'\n');self.events.flush()
    def save_raw(self,image,name):
        # Encode original pixels before any display overlay. Preserve unusual geometry for diagnosis.
        if not isinstance(image,np.ndarray) or image.dtype!=np.uint8: raise ValueError('Raw capture is not a uint8 image')
        ok,encoded=cv2.imencode('.png',image)
        if not ok: raise ValueError('Raw PNG encoding failed')
        blob=encoded.tobytes();path=self.out/'raw'/name
        with path.open('xb') as f:f.write(blob)
        return str(path),sha_bytes(blob)
    def set_reference(self,image,path,digest):
        if self.reference is not None: raise ValueError('Fixed reference cannot be replaced during a session')
        self.reference=checked_image(image).copy();self.reference.setflags(write=False);self.refpath=path;self.refhash=digest
        self.sources[Path(path).resolve()]=digest
        self.event('FIXED_UNLOADED_REFERENCE',path=path,sha256=digest,user_declared=True)
    def base_row(self,index,stamp,wall,read_ms):
        row={k:'' for k in FIELDS};row.update(attempt_index=index,frame_id=f'frame_{index:06d}',timestamp_ns=stamp,
            host_wall_time_ns=wall,timestamp_kind='host_receive_monotonic_ns_NOT_exposure',sensor_id=self.sensor,model=self.model.name,
            valid=0,normal_sign='' if self.sign is None else self.sign,tare_offset_raw_n='' if self.tare is None else self.tare,
            tare_event_timestamp_ns='' if self.tare_time is None else self.tare_time,reference_path=self.refpath,
            reference_sha256=self.refhash,capture_read_ms=read_ms,status='INVALID')
        row.update(identity_status='UNRESOLVED',capture_snapshot_sensor_id=CAPTURE_SENSOR,inherited_profile_sensor_id=INHERITED_SENSOR,
                   calibration_scope='CURRENT_FAMA_SENSOR_0_100G')
        return row
    def process(self,row,image):
        started=time.perf_counter()
        try:
            if self.reference is None: raise ValueError('No fixed unloaded reference')
            checked_image(image)
            a=time.perf_counter();fields=self.frontend.process(self.reference,image);row['frontend_ms']=(time.perf_counter()-a)*1000
            row['quality_valid_fraction']=fields['quality_valid_fraction'];row['confidence_mean']=float(fields['confidence'].mean())
            if fields['quality_valid_fraction']<.2: raise ValueError('Valid grid fraction below 20%; prediction unavailable')
            a=time.perf_counter();fz=float(self.model.predict(fields['uv'],fields['valid'])[0]);row['model_ms']=(time.perf_counter()-a)*1000
            if not np.isfinite(fz): raise ValueError('Nonfinite Fz prediction')
            quality_status='VALID_RAW_FZ' if fields['quality_valid_fraction']>=.9 else 'DEGRADED_RAW_FZ'
            warning=DEMO_RANGE_WARNING if fz<DEMO_OUTPUT_RANGE[0] or fz>DEMO_OUTPUT_RANGE[1] else ''
            row.update(fz_raw_n=fz,valid=1,quality_status=quality_status,status=warning or quality_status,demo_output_range_warning=warning,
                       display_fz_n=(fz-(self.tare or 0.0))*(self.sign if self.sign is not None else 1))
        except Exception as exc: row.update(valid=0,fz_raw_n='',display_fz_n='',status='INVALID',error=str(exc))
        row['processing_ms']=(time.perf_counter()-started)*1000
        return row
    def write(self,row):
        self.writer.writerow(row);self.file.flush();self.rows+=1
        self.history.append(dict(row))
        threshold=row['timestamp_ns']-30_000_000_000
        self.history=[r for r in self.history if r['timestamp_ns']>=threshold]
        if row['status']!='REFERENCE_CAPTURED' and not row['valid']:self.invalid+=1
        if row.get('demo_output_range_warning'):self.range_warnings+=1
    def request_tare(self,row):
        if self.tare is not None: self.event('TARE_REJECTED_ALREADY_SET',source_attempt=row['attempt_index']);return False
        if not row['valid']:self.event('TARE_REJECTED_INVALID_FRAME',source_attempt=row['attempt_index']);return False
        self.tare=float(row['fz_raw_n']);self.tare_time=time.perf_counter_ns()
        self.event('USER_ONE_TIME_TARE',offset_raw_fz_n=self.tare,source_attempt=row['attempt_index'],source_timestamp_ns=row['timestamp_ns'],
                   tare_timestamp_ns=self.tare_time,effective_from_attempt=row['attempt_index']+1)
        return True
    def finish(self,status,error=''):
        changed=any(not p.is_file() or sha_file(p)!=h for p,h in self.sources.items())
        if changed: status='FAIL_SOURCE_CHANGED';error=(error+'; runtime/model/frontend files changed').strip('; ')
        elif self.invalid and status in ('COMPLETE_REPLAY','COMPLETE_LIMIT','STOPPED_BY_USER'):status='COMPLETE_WITH_INVALID_ROWS'
        self.file.close();self.events.close()
        receipt=dict(status=status,readiness='NOT_READY',rows=self.rows,invalid_rows=self.invalid,output_range_warning_rows=self.range_warnings,error=error,
            identity_status='UNRESOLVED',declared_sensor_id=self.sensor,capture_snapshot_sensor_id=CAPTURE_SENSOR,inherited_profile_sensor_id=INHERITED_SENSOR,
            calibration_scope='CURRENT_FAMA_SENSOR_0_100G',validation_status='RESUBSTITUTION_FIT_AND_SHARED_REFERENCE_CYCLE_DIAGNOSTICS_ONLY',
            range_check_meaning='OUTPUT_WARNING_ONLY_NOT_OOD_OR_CONFIDENCE',
            sources_unchanged=not changed,mode=self.mode,capture_mode=self.capture_mode,capture_provider=self.capture_provider,
            camera_requested=self.capture_mode=='USER_OPENCV_CAPTURE',backend_opened=self.backend_opened,
            camera_accessed=self.capture_mode=='USER_OPENCV_CAPTURE' and self.backend_opened,
            physical_accuracy='UNVERIFIED',exposure_timestamps_available=False,raw_predictions_preserved=True,
            hashes={p.name:sha_file(p) for p in [self.out/'session.json',self.out/'predictions.csv',self.out/'events.jsonl']})
        save_json(self.out/'receipt.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False));return 0 if status in ('COMPLETE_REPLAY','COMPLETE_LIMIT','STOPPED_BY_USER') and self.invalid==0 and not changed else 2

def output_dir(value):
    return Path(value) if value else ROOT/'recordings'/('session_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])

def replay(args,model=None,frontend=None):
    declare(args.sensor_id,args.declare_unloaded)
    model=model or ForceModel(args.model);frontend=frontend or PairFrontend()
    session=Session(output_dir(args.output),'saved_image_replay',args.sensor_id,model,frontend,args.normal_sign)
    try:
        reference_error=''
        try:
            reference,digest=read_image(args.reference);path=Path(args.reference).resolve()
            session.set_reference(reference,str(path),digest);session.sources[path]=digest
        except Exception as exc:reference_error='REFERENCE_ERROR: '+str(exc)
        for index,source in enumerate(args.images):
            start=time.perf_counter();stamp=time.perf_counter_ns();wall=time.time_ns()
            row=session.base_row(index,stamp,wall,0)
            try:
                image,digest=read_image(source)
                row.update(timestamp_ns=time.perf_counter_ns(),host_wall_time_ns=time.time_ns(),
                    raw_image_path=str(Path(source).resolve()),raw_image_sha256=digest,capture_read_ms=(time.perf_counter()-start)*1000)
                session.sources[Path(source).resolve()]=digest
                if reference_error:raise ValueError(reference_error)
                session.process(row,image)
            except Exception as exc:
                row.update(raw_image_path=str(Path(source).resolve()),error=str(exc),status='INVALID',capture_read_ms=(time.perf_counter()-start)*1000)
            session.write(row)
        return session.finish('COMPLETE_REPLAY' if session.invalid==0 else 'COMPLETE_WITH_INVALID_ROWS')
    except Exception as exc:return session.finish('FAIL_REPLAY',str(exc))

def open_camera(index,backend):
    api={'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF,'any':cv2.CAP_ANY}[backend]
    capture=cv2.VideoCapture(index,api)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,640);capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    return capture

def history_segments(history,column,end_ns,window_seconds=30):
    """Each point is a logged raw/display value. Invalid rows explicitly split lines."""
    segments=[];current=[]
    for row in history:
        if row['timestamp_ns']<end_ns-window_seconds*1e9:continue
        value=row.get(column,'')
        if not row['valid'] or value=='' or not np.isfinite(float(value)):
            if current:segments.append(current);current=[]
            continue
        current.append(((row['timestamp_ns']-end_ns)/1e9,float(value)))
    if current:segments.append(current)
    return segments

def show(image,row,session):
    canvas=np.zeros((720,960,3),np.uint8)
    if isinstance(image,np.ndarray) and image.shape==(480,640,3):canvas[:240,:320]=cv2.resize(image,(320,240))
    lines=['CALIBRATION FIT ONLY | NOT_READY | target 0-100g',f'{session.model.name} | attempt {row["attempt_index"]}',
           'Raw signed Fz: '+(f'{row["fz_raw_n"]:.4f} N' if row['valid'] else 'UNAVAILABLE'),
           'Display: '+(f'{row["display_fz_n"]:.4f} N' if row['valid'] else 'UNAVAILABLE'),
           f'valid={row["quality_valid_fraction"]} sign={session.sign} tare={session.tare}',
           'Identity UNRESOLVED: capture 000015 / inherited 000011',
           'T: one-time tare | Q / Esc: stop',row.get('demo_output_range_warning','') or row['error'][:80]]
    for i,line in enumerate(lines):cv2.putText(canvas,line,(330,23+i*29),cv2.FONT_HERSHEY_SIMPLEX,.46,(0,100,255) if line==DEMO_RANGE_WARNING else (0,255,255),1,cv2.LINE_AA)
    column='display_fz_n' if session.sign is not None or session.tare is not None else 'fz_raw_n'
    segments=history_segments(session.history,column,row['timestamp_ns'])
    values=[value for segment in segments for _,value in segment]
    lo=min(values) if values else -1.;hi=max(values) if values else 1.
    margin=max(.05,(hi-lo)*.1);lo-=margin;hi+=margin
    left,top,right,bottom=75,305,930,665
    cv2.putText(canvas,column+' [N] | host receive time; no smoothing; invalid = gap',(75,275),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
    cv2.rectangle(canvas,(left,top),(right,bottom),(130,130,130),1)
    for i in range(5):
        value=lo+(hi-lo)*i/4;y=int(bottom-(bottom-top)*i/4)
        cv2.line(canvas,(left,y),(right,y),(45,45,45),1)
        cv2.putText(canvas,f'{value:.3f}',(5,y+5),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
    for seconds in (-30,-20,-10,0):
        x=int(left+(seconds+30)/30*(right-left));cv2.putText(canvas,str(seconds)+' s',(x-15,bottom+25),cv2.FONT_HERSHEY_SIMPLEX,.45,(200,200,200),1)
    for segment in segments:
        points=np.array([[int(left+(t+30)/30*(right-left)),int(bottom-(v-lo)/(hi-lo)*(bottom-top))] for t,v in segment],np.int32)
        if len(points)>1:cv2.polylines(canvas,[points],False,(0,220,130),1,cv2.LINE_8)
        elif len(points)==1:cv2.circle(canvas,tuple(points[0]),2,(0,220,130),-1)
    cv2.imshow('CALIBRATION FIT ONLY - NOT_READY - target 0-100g',canvas)
    return cv2.waitKey(1)&0xff

def live(args,model=None,frontend=None,capture_factory=None,key_reader=None):
    declare(args.sensor_id,args.declare_unloaded)
    if type(args.camera_index) is not int or args.camera_index<0: raise ValueError('Explicit nonnegative camera index required')
    if not 0<args.duration_seconds<=3600: raise ValueError('duration-seconds must be in (0,3600]')
    model=model or ForceModel(args.model);frontend=frontend or PairFrontend()
    session=Session(output_dir(args.output),'live',args.sensor_id,model,frontend,args.normal_sign,
                    capture_mode='SIMULATED_TEST' if capture_factory is not None else 'USER_OPENCV_CAPTURE')
    session.event('CAMERA_REQUEST',index=args.camera_index,backend=args.backend,duration_seconds=args.duration_seconds,
                  reference_policy='first_successful_exact_shape_frame_after_explicit_user_unloaded_declaration')
    capture=None;status='COMPLETE_LIMIT';error='';started=time.perf_counter();index=0;consecutive_failures=0
    try:
        capture=(capture_factory or open_camera)(args.camera_index,args.backend)
        session.backend_opened=bool(capture.isOpened())
        if not session.backend_opened: raise RuntimeError('Cannot open explicitly selected camera')
        while time.perf_counter()-started<args.duration_seconds and (args.max_frames is None or index<args.max_frames):
            image=None;before=time.perf_counter();read_error=''
            try:ok,image=capture.read()
            except Exception as exc:ok=False;read_error=repr(exc)
            row=session.base_row(index,time.perf_counter_ns(),time.time_ns(),(time.perf_counter()-before)*1000)
            try:
                if not ok or image is None: raise RuntimeError(read_error or 'Camera read failed; no hold-last')
                a=time.perf_counter();path,digest=session.save_raw(image,f'frame_{index:06d}.png');row.update(raw_image_path=path,raw_image_sha256=digest,raw_save_ms=(time.perf_counter()-a)*1000)
                checked_image(image)
                if session.reference is None:
                    session.set_reference(image,path,digest)
                    row.update(status='REFERENCE_CAPTURED',reference_path=path,reference_sha256=digest)
                else: session.process(row,image)
                consecutive_failures=0 if row['status']=='REFERENCE_CAPTURED' or row['valid'] else consecutive_failures+1
            except Exception as exc:row.update(error=str(exc),status='INVALID');consecutive_failures+=1
            session.write(row);index+=1
            if key_reader is not None:key=key_reader(image,row,session)
            elif not args.no_window:key=show(image,row,session)
            else:key=-1
            if key in (ord('t'),ord('T')):session.request_tare(row)
            if key in (ord('q'),ord('Q'),27):status='STOPPED_BY_USER';break
            if consecutive_failures>=5:status='FAIL_CONSECUTIVE_INVALID';error='Five consecutive capture/geometry/quality/inference failures';break
        if session.reference is None:status='FAIL_NO_REFERENCE';error='No usable explicitly declared unloaded reference'
        elif session.rows<2:status='FAIL_NO_FORCE_FRAME';error='No current frame after reference capture'
    except KeyboardInterrupt:status='INTERRUPTED';error='User interrupted acquisition'
    except Exception as exc:status='FAIL_CAPTURE';error=str(exc)
    finally:
        try:
            if capture is not None:capture.release()
            if not args.no_window:cv2.destroyAllWindows()
        except Exception as exc:status='FAIL_CLEANUP';error=str(exc)
    return session.finish(status,error)

def offline_check(args):
    frontend=PairFrontend();rng=np.random.default_rng(17)
    image=rng.integers(20,230,(480,640,3),dtype=np.uint8)
    result=frontend.process(image,image)
    reports=[]
    for name in MODELS:
        model=ForceModel(name);output=model.predict(result['uv'],result['valid']);model.assert_unchanged()
        reports.append(dict(model=name,synthetic_raw_fz_n=float(output[0])))
    frontend.assert_unchanged()
    print(json.dumps(dict(status='PASS_OFFLINE_IMPORT_GRAPH_AND_FRONTEND_ONLY',readiness='NOT_READY',camera_accessed=False,synthetic=True,models=reports)))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('offline-check',help='Load all models using synthetic pixels; never opens a camera')
    for command in ('replay','live'):
        p=commands.add_parser(command);p.add_argument('--sensor-id',required=True);p.add_argument('--declare-unloaded',action='store_true',required=True)
        p.add_argument('--model',choices=MODELS,default='lowmass_head60');p.add_argument('--output');p.add_argument('--normal-sign',type=int,choices=(-1,1))
        if command=='replay':p.add_argument('--reference',required=True);p.add_argument('--images',nargs='+',required=True)
        else:
            p.add_argument('--camera-index',type=int,required=True);p.add_argument('--backend',choices=('dshow','msmf','any'),default='dshow')
            p.add_argument('--duration-seconds',type=float,default=300);p.add_argument('--max-frames',type=int);p.add_argument('--no-window',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='offline-check':return offline_check(args)
        if getattr(args,'max_frames',None) is not None and args.max_frames<2:raise ValueError('max-frames must be at least 2, including reference')
        return replay(args) if args.command=='replay' else live(args)
    except Exception as exc:print('PLANAR_FORCE_FAILED:',repr(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
