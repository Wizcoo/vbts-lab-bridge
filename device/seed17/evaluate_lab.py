"""Summarize original marked plateaus; no filtering, time shifting, or refitting."""
import csv,datetime,html,json,math,uuid
from pathlib import Path
import numpy as np

def store(path,rows,keys):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def graph(rows):
    width,height=970,290; left,top,right,bottom=58,32,945,245
    vals=[r[k] for r in rows for k in ('press_N','meter_press_N') if r[k] is not None]
    lo=min([0,*vals]);hi=max([1,*vals]);tm=max(1e-9,rows[-1]['t_s'])
    Y=lambda v:bottom-(bottom-top)*(v-lo)/(hi-lo)
    X=lambda t:left+(right-left)*t/tm
    parts=[f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg"><rect width="100%" height="100%" fill="white"/>']
    for v in np.linspace(lo,hi,5):
        parts.append(f'<path d="M{left} {Y(v):.2f}H{right}" stroke="#ddd"/><text x="2" y="{Y(v):.2f}">{v:.2f}</text>')
    for key,col in [('meter_press_N','#273746'),('press_N','#007e79')]:
        path=[];previous=None
        for r in rows:
            v=r[key]
            if v is None:previous=None;continue
            contiguous=previous is not None and r['frame_seq']==previous['frame_seq']+1 and r['ref_id']==previous['ref_id'] and r['epoch']==previous['epoch']
            path.append(('L' if contiguous else 'M')+f"{X(r['t_s']):.2f} {Y(v):.2f}")
            if not contiguous:parts.append(f'<circle cx="{X(r["t_s"]):.2f}" cy="{Y(v):.2f}" r="1.4" fill="{col}"/>')
            previous=r
        parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{col}" stroke-width="1.5"/>')
    parts.append(f'<text x="{left}" y="20" fill="#273746">FT (N)</text><text x="{left+88}" y="20" fill="#007e79">Seed 17 (N)</text><text x="{left}" y="274">0 s</text><text x="{right-80}" y="274">{tm:.2f} s</text></svg>')
    return ''.join(parts)

def evaluate(session):
    session=Path(session);source=session/'capture_sync.csv'
    with source.open(encoding='utf-8-sig',newline='') as f:stream=list(csv.DictReader(f))
    if not stream:raise ValueError('No recorded frame pairs. Record with R before making a report.')
    rows=[];groups={};t0=int(stream[0]['grab_end_ns'])
    for s in stream:
        pred=json.loads(s['prediction_json'] or 'null');ex=json.loads(s['extra_json'] or '{}')
        if not isinstance(pred,dict) or pred.get('model')!='uv_mlp_17':raise ValueError('session lacks seed17 predictions')
        val=pred.get('press_N') if pred.get('status')=='OK' else None
        ft=-float(s['Fz']) if s['label_valid'].lower() in ('true','1') and s['Fz'] else None
        err=val-ft if val is not None and ft is not None else None
        r={'frame_seq':int(s['frame_seq']),'t_s':(int(s['grab_end_ns'])-t0)/1e9,'ref_id':s['reference_id'],'epoch':s['tare_epoch'],
           'marker':ex.get('plateau_id',0),'target_N':ex.get('plateau_target_N'),'press_N':val,'meter_press_N':ft,'error_N':err,
           'prediction_status':pred.get('status'),'pair_status':s['sync_status'],'image':s['image_filename']}
        rows.append(r)
        if r['target_N'] is not None:groups.setdefault((r['ref_id'],r['epoch'],r['marker'],r['target_N']),[]).append(r)
    summary=[]
    keys=['reference','epoch','marker','target_N','frames','valid_pairs','duration_s','meter_mean_N','meter_span_N','pred_mean_N','pred_span_N','bias_N','mae_N','p95_abs_N','max_abs_N','late_minus_early_error_N']
    for (ref,epoch,mark,target),items in groups.items():
        good=[r for r in items if r['error_N'] is not None]
        row=dict(reference=ref,epoch=epoch,marker=mark,target_N=target,frames=len(items),valid_pairs=len(good),duration_s=items[-1]['t_s']-items[0]['t_s'])
        if good:
            m=np.array([r['meter_press_N'] for r in good]);p=np.array([r['press_N'] for r in good]);e=p-m
            times=np.array([r['t_s'] for r in good]);begin=items[0]['t_s'];dur=row['duration_s']
            early=e[times<=begin+.2*dur];late=e[times>=begin+.8*dur]
            row.update(meter_mean_N=m.mean(),meter_span_N=np.ptp(m),pred_mean_N=p.mean(),pred_span_N=np.ptp(p),bias_N=e.mean(),mae_N=np.abs(e).mean(),p95_abs_N=np.quantile(np.abs(e),.95),max_abs_N=max(abs(e)),
                       late_minus_early_error_N=late.mean()-early.mean() if len(good)>=5 and len(early) and len(late) and dur>0 else None)
        summary.append(row)
    out=session.parent/(session.name+'_seed17_report_'+uuid.uuid4().hex[:8]);out.mkdir(exist_ok=False)
    store(out/'frames.csv',rows,rows[0].keys());store(out/'plateaus.csv',summary,keys)
    def cell(v):return '—' if v is None else f'{v:.3f}' if isinstance(v,float) else html.escape(str(v))
    grid='<table><tr>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(r.get(k))+'</td>' for k in keys)+'</tr>' for r in summary)+'</table>'
    msg='已标记平台按真实配对力计 Fz 计算。' if summary else '没有标记平台。本次仅可查看逐帧曲线。'
    (out/'report.html').write_text('<!doctype html><meta charset=utf-8><title>种子17 实测</title><style>body{font:15px sans-serif;margin:24px}svg{max-width:1100px;width:100%}th,td{padding:6px;border-bottom:1px solid #ddd}</style><h1>实测力阶梯</h1><p>'+msg+' 按压为正。无效配对留空，图像和仪器均为主机时间，不能由此认证动态延迟。</p>'+graph(rows)+grid,encoding='utf8')
    return out
