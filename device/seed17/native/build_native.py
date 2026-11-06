"""Optional rebuild in an x64 MSVC developer shell using the matching internal SDK."""
import argparse,hashlib,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sdk-repo',type=Path,required=True)
parser.add_argument('--opencv-build',type=Path,required=True,help='OpenCV 4.4 root with include/ and x64/vc15/lib/')
parser.add_argument('--output',type=Path,required=True,help='new isolated directory')
args=parser.parse_args()
here=Path(__file__).resolve().parent
manifest=json.loads((here.parent/'manifest.json').read_text())
original=(args.sdk_repo/'src/VT_Tracker.cpp').read_text(encoding='utf-8-sig')
anchor='ws.dis->calc(grayRef, grayCur, ws.flow); //计算得到每个像素点的位移'
if original.count(anchor)!=1:raise RuntimeError('SDK source revision does not match the validated patch')
newdir=args.output.resolve();newdir.mkdir(parents=True,exist_ok=False)
patched=newdir/'VT_Tracker.cpp'
patched.write_text(original.replace(anchor,'ws.flow.setTo(cv::Scalar(0, 0));\n\t\t\t'+anchor),encoding='utf8',newline='\r\n')
if hashlib.sha256(patched.read_bytes()).hexdigest()!=manifest['tracker_source_sha256']:
    raise RuntimeError('Patched source identity differs from the validated candidate')
cv=args.opencv_build.resolve()
subprocess.run(['cl','/nologo','/LD','/MD','/O2','/EHsc','/utf-8','/std:c++17','/openmp','/DNDEBUG','/DVT_TRACKER_STATIC',
                '/I'+str(args.sdk_repo.resolve()/'inc'),'/I'+str(cv/'include'),str(here/'bridge.cpp'),str(patched),
                '/link','/OUT:seed17_tracker.dll',str(cv/'x64/vc15/lib/opencv_world440.lib')],cwd=newdir,check=True)
print('Rebuilt DLL SHA256:',hashlib.sha256((newdir/'seed17_tracker.dll').read_bytes()).hexdigest())
print('A new binary needs its own manifest and parity review before use.')
