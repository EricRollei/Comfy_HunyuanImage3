@echo off
setlocal EnableExtensions
chcp 65001 >nul
title HunyuanImage 3 - Fix Accelerate and Transformers

rem This BAT must be placed in the ComfyUI root folder:
rem C:\ComfyUI\fix_hunyuan_python_versions.bat
rem Expected embedded Python:
rem C:\ComfyUI\python_embeded\python.exe

set "ROOT=%~dp0"
set "PYTHON=%ROOT%python_embeded\python.exe"
set "BACKUP=%ROOT%packages_before_hunyuan_fix.txt"

echo ============================================================
echo HunyuanImage 3 dependency repair
echo ============================================================
echo.
echo Required versions:
echo   accelerate   1.14.0
echo   transformers 4.57.6
echo.

if not exist "%PYTHON%" (
    echo [ERROR] Embedded Python was not found:
    echo         "%PYTHON%"
    echo.
    echo Put this BAT in the ComfyUI root folder, for example:
    echo C:\ComfyUI\fix_hunyuan_python_versions.bat
    goto :failed
)

echo [1/5] Saving the current package list...
"%PYTHON%" -m pip freeze > "%BACKUP%"
if errorlevel 1 (
    echo [WARNING] Could not save the package list.
) else (
    echo       Saved to:
    echo       "%BACKUP%"
)
echo.

echo [2/5] Removing existing Accelerate and Transformers...
"%PYTHON%" -m pip uninstall -y accelerate transformers
if errorlevel 1 goto :failed
echo.

echo [3/5] Installing the tested versions...
"%PYTHON%" -m pip install --no-cache-dir "accelerate==1.14.0" "transformers==4.57.6"
if errorlevel 1 goto :failed
echo.

echo [4/5] Verifying imports and versions...
"%PYTHON%" -c "import accelerate, transformers; print('accelerate:', accelerate.__version__); print('transformers:', transformers.__version__); assert accelerate.__version__ == '1.14.0'; assert transformers.__version__ == '4.57.6'; print('VERSION CHECK: OK')"
if errorlevel 1 goto :failed
echo.

echo [5/5] Checking the main Hunyuan imports...
"%PYTHON%" -c "from transformers import AutoModelForCausalLM; from accelerate import init_empty_weights; print('HUNYUAN IMPORT CHECK: OK')"
if errorlevel 1 goto :failed
echo.

echo ============================================================
echo SUCCESS
echo ============================================================
echo Installed:
echo   accelerate   1.14.0
echo   transformers 4.57.6
echo.
echo Completely close and restart ComfyUI before generating.
echo Do not run Update All after this unless you are ready to
echo restore these versions again.
echo.
pause
exit /b 0

:failed
echo.
echo ============================================================
echo INSTALLATION FAILED
echo ============================================================
echo Review the messages above.
echo The previous package list, if created, is here:
echo "%BACKUP%"
echo.
pause
exit /b 1
