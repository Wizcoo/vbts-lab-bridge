param(
    [Parameter(Mandatory=$true)][string]$SdkRoot,
    [string]$Python = 'python',
    [int]$Camera = 0,
    [string]$Port = '',
    [string]$Specimen,
    [string]$Position,
    [switch]$Capture
)
$ErrorActionPreference = 'Stop'
$base = (Resolve-Path -LiteralPath $SdkRoot).Path
$candidates = @($base, (Join-Path $base 'common/bin'), (Join-Path $base 'sdk_dist_identity/common/bin'), (Join-Path $base 'x64/Release'))
$bin = $candidates | Where-Object {Test-Path -LiteralPath (Join-Path $_ 'opencv_world440.dll')} | Select-Object -First 1
if (-not $bin) {throw 'Pass the SDK, repository, or bin directory containing opencv_world440.dll.'}
$cv = Join-Path $bin 'opencv_world440.dll'
$cli = Join-Path $PSScriptRoot 'lab.py'
& $Python $cli selftest --opencv-dll $cv
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
if ($Capture) {
    if (-not $Position -or -not $Specimen) {throw 'Capture needs -Position and -Specimen.'}
    $ft = Join-Path $bin 'hps_ft.dll'
    if (-not (Test-Path -LiteralPath $ft)) {throw 'hps_ft.dll is missing. Specify --ft-dll with lab.py capture for a custom location.'}
    $arguments = @($cli,'capture','--opencv-dll',$cv,'--ft-dll',$ft,'--profile',(Join-Path $PSScriptRoot 'HSMVBTS_G01_H20_000011.txt'),'--camera',"$Camera",'--position',$Position,'--specimen',$Specimen)
    if ($Port) {$arguments += @('--port',$Port)}
    & $Python @arguments
    exit $LASTEXITCODE
}
