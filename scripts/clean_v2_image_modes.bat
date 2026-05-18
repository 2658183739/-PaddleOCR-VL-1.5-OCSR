@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..
set PYTHON_BIN=python
set DRY=
if "%1"=="--dry-run" set DRY=--dry-run

cd /d "%PROJECT_ROOT%"
%PYTHON_BIN% V2\scripts\normalize_image_modes.py ^
  --root "%PROJECT_ROOT%\V2\data\assets\train_phase1" ^
  --root "%PROJECT_ROOT%\V2\data\assets\train_phase2" ^
  --root "%PROJECT_ROOT%\V2\data\assets\train_phase3" ^
  --root "%PROJECT_ROOT%\V2\data\eval\canonical_smiles_main_v1\images" ^
  %DRY%
