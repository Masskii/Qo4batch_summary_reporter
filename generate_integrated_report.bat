@echo off
REM Integrated Report Generation Script for 4-Batch CNN Tool
REM This script automates the complete process of generating integrated_report.html
REM
REM Author: 4-Batch CNN Tool Team
REM Date: 2026-03-06
REM

setlocal enabledelayedexpansion

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"

REM Configuration
set "BASE_DIR=%BASE_DIR%."
set "IMAGE_DIR=%IMAGE_DIR%..\..\4batch_input\image"
set "HEADS_DIR=%HEADS_DIR%.\heads"
set "OUTPUT_FILE=%OUTPUT_FILE%integrated_report.html"
set "BATCH_SIZE=%BATCH_SIZE%4"

REM Print banner
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║   Integrated Report Generation Script for 4-Batch CNN Tool     ║
echo ║   自动生成 integrated_report.html 的完整流程脚本                 ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Step 1: Check Python environment
echo [STEP 1/6] Checking Python environment
echo ─────────────────────────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo ✗ Python not found! Please install Python 3.7+ first.
    echo   Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM Step 2: Check required directories
echo [STEP 2/6] Checking required directories
echo ─────────────────────────────────────────────────────────────────────
if not exist "%BASE_DIR%" (
    echo ✗ Base directory not found: %BASE_DIR%
    pause
    exit /b 1
)
echo ✓ Base directory: %BASE_DIR%

REM Count model directories
set /a MODEL_COUNT=0
for /f %%d in ('dir /b /ad "%BASE_DIR%" 2^>nul ^| findstr /v "^\." ^| findstr /v "__pycache__" ^| findstr /v ".claude"') do (
    set /a MODEL_COUNT+=1
)
if !MODEL_COUNT! EQU 0 (
    echo ✗ No model directories found in %BASE_DIR%
    pause
    exit /b 1
)
echo ✓ Found !MODEL_COUNT! model directories
echo.

REM Step 3: Check for postprocess data
echo [STEP 3/6] Checking for existing postprocess data
echo ─────────────────────────────────────────────────────────────────────
set "HAS_POSTPROCESS=0"
for /d %%d in ("%BASE_DIR%\*") do (
    if exist "%%~d\postprocess" (
        set "HAS_POSTPROCESS=1"
        goto :found_postprocess
    )
)
:found_postprocess

if "!HAS_POSTPROCESS!" EQU "0" (
    echo ℹ No postprocess data found. You may need to run postprocessing first.
    echo   Run: python 4batch_postprocess_all.py --base-dir . --image-dir ..\..\4batch_input\image --heads-dir .\heads
    echo.
    set /p RUN_POSTPROCESS="Do you want to run postprocessing now? (y/N): "
    if /i "!RUN_POSTPROCESS!" EQU "y" (
        echo.
        echo [STEP 3b/6] Running postprocessing
        echo ─────────────────────────────────────────────────────────────────────
        if not exist "%IMAGE_DIR%" (
            echo ✗ Image directory not found: %IMAGE_DIR%
            pause
            exit /b 1
        )
        if not exist "%HEADS_DIR%" (
            echo ✗ Heads directory not found: %HEADS_DIR%
            pause
            exit /b 1
        )
        python "%SCRIPT_DIR%\4batch_postprocess_all.py" --base-dir "%BASE_DIR%" --image-dir "%IMAGE_DIR%" --heads-dir "%HEADS_DIR%" --batch-size %BATCH_SIZE%
        if errorlevel 1 (
            echo ✗ Postprocessing failed!
            pause
            exit /b 1
        )
        echo ✓ Postprocessing completed
    ) else (
        echo ℹ Skipping postprocessing. Proceeding with existing data...
    )
) else (
    echo ✓ Postprocess data found
)
echo.

REM Step 4: Check for profile_core files (performance data)
echo [STEP 4/6] Checking for performance data
echo ─────────────────────────────────────────────────────────────────────
set "HAS_PROFILE=0"
for /d %%d in ("%BASE_DIR%\*") do (
    if exist "%%~d\profile_core0.json" (
        set "HAS_PROFILE=1"
        goto :found_profile
    )
)
:found_profile

if "!HAS_PROFILE!" EQU "0" (
    echo ℹ No profile_core files found. Performance data will not be included in the report.
) else (
    echo ✓ Performance data found
)
echo.

REM Step 5: Generate the integrated report
echo [STEP 5/6] Generating integrated report
echo ─────────────────────────────────────────────────────────────────────
python "%SCRIPT_DIR%\generate_integrated_report.py" -d "%BASE_DIR%" -o "%OUTPUT_FILE%"

if errorlevel 1 (
    echo ✗ Report generation failed!
    pause
    exit /b 1
)
echo ✓ Report generated successfully
echo.

REM Step 6: Display report location
echo [STEP 6/6] Report generated
echo ─────────────────────────────────────────────────────────────────────
echo.
echo ╔════════════════════════════════════════════════════════════════╗
echo ║   Report Generation Complete!                                 ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
set "FULL_PATH=%SCRIPT_DIR%\%OUTPUT_FILE%"
echo Report location: %FULL_PATH%
echo.
echo To view the report, open the file in your browser:
echo   File: %OUTPUT_FILE%
echo.

REM Ask if user wants to open the report
set /p OPEN_REPORT="Do you want to open the report in your browser? (y/N): "
if /i "%OPEN_REPORT%" EQU "y" (
    start "" "%FULL_PATH%"
    echo ✓ Opening report in browser...
)
echo.
echo For more information, see integrated_report_readme.md
echo.
timeout /t 3 >nul
