@echo off
setlocal
rem No PowerShell, policy changes, automatic installation or camera-index default.
rem PLANAR_FORCE_PYTHON is an executable path without embedded quotation marks.
if defined PLANAR_FORCE_PYTHON goto explicit_python
if exist "%~dp0.venv\Scripts\python.exe" goto local_python
python -B -X utf8 "%~dp0planar_force.py" live %*
exit /b %errorlevel%

:explicit_python
"%PLANAR_FORCE_PYTHON%" -B -X utf8 "%~dp0planar_force.py" live %*
exit /b %errorlevel%

:local_python
"%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0planar_force.py" live %*
exit /b %errorlevel%
