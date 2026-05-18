@echo off
setlocal
set PROJECT_ROOT=%~dp0..\..

for %%D in (phase1_lora phase2_lora phase3_lora) do (
  if exist "%PROJECT_ROOT%\V2\outputs\%%D" rmdir /s /q "%PROJECT_ROOT%\V2\outputs\%%D"
  mkdir "%PROJECT_ROOT%\V2\outputs\%%D"
  echo reset %PROJECT_ROOT%\V2\outputs\%%D
)
