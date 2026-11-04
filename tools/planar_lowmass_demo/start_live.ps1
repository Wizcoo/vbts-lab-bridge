param(
    [Parameter(Mandatory=$true)][int]$CameraIndex,
    [Parameter(Mandatory=$true)][string]$SensorId,
    [Parameter(Mandatory=$true)][switch]$DeclareUnloaded,
    [string]$Python = 'python',
    [ValidateSet('lowmass_head60','lowmass_aug60')]
    [string]$Model = 'lowmass_head60',
    [ValidateSet('dshow','msmf','any')][string]$Backend = 'dshow',
    [double]$DurationSeconds = 300,
    [string]$Output,
    [ValidateSet(-1,1)][int]$NormalSign
)
$ErrorActionPreference = 'Stop'
if (-not $DeclareUnloaded) { throw 'Keep the sensor completely unloaded and explicitly pass -DeclareUnloaded.' }
$arguments = @('-B','-X','utf8',(Join-Path $PSScriptRoot 'planar_force.py'),'live','--camera-index',[string]$CameraIndex,'--sensor-id',$SensorId,'--declare-unloaded','--model',$Model,'--backend',$Backend,'--duration-seconds',[string]$DurationSeconds)
if ($Output) { $arguments += @('--output',$Output) }
if ($PSBoundParameters.ContainsKey('NormalSign')) { $arguments += @('--normal-sign',[string]$NormalSign) }
& $Python @arguments
exit $LASTEXITCODE
