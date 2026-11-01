@echo off
rem Issue #17 operator script: check (read-only) or rebuild-for-comparison the scope30-v1 report and PDF.
rem
rem   run-issue17-report.bat CHECK [expected-commit]   read-only: reconcile docs\REPORT.md and docs\REPORT.pdf
rem                                                    with the accepted evidence and run the report tests
rem   run-issue17-report.bat BUILD [expected-commit]   CHECK, then rebuild the PDF OUTSIDE the repository
rem                                                    (.scratch\issue17\) and compare it with docs\REPORT.pdf
rem
rem Run it from a clean checkout of the reviewed PR head or of main after the merge. Pass the commit you
rem expect (for example the PR head SHA) to make the script refuse any other checkout.
rem Set PYTHON to choose an interpreter (default: py -3.11 if available, otherwise python).
rem The script stops at the first failure and writes a log to .scratch\issue17\.
rem It never runs generation, Ollama, promotion, retraining, freezing, release tagging, presentation
rem work or --write-pdf, and it never modifies tracked files.

setlocal EnableExtensions
cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=CHECK"
if /I not "%MODE%"=="CHECK" if /I not "%MODE%"=="BUILD" (
  echo Usage: run-issue17-report.bat CHECK^|BUILD [expected-commit]
  exit /b 2
)
set "EXPECTED=%~2"

if not exist ".scratch\issue17" mkdir ".scratch\issue17"
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set "STAMP=%%t"
set "LOG=%CD%\.scratch\issue17\report-%MODE%-%STAMP%.log"
echo Issue #17 report %MODE% - log: %LOG%
echo Issue #17 report %MODE% started %DATE% %TIME% > "%LOG%"

if "%PYTHON%"=="" (
  py -3.11 --version >nul 2>&1 && (set "PYTHON=py -3.11") || (set "PYTHON=python")
)

call :step "Repository revision" git rev-parse HEAD || goto :fail
for /f %%h in ('git rev-parse HEAD') do set "HEAD=%%h"
if not "%EXPECTED%"=="" (
  echo %HEAD% | findstr /B /I "%EXPECTED%" >nul || (
    echo FAIL checkout is %HEAD%, expected %EXPECTED% >> "%LOG%"
    echo FAIL checkout is %HEAD%, expected %EXPECTED%
    goto :fail
  )
)
rem Tracked files must be unchanged so the check describes committed, reviewed content.
git diff --quiet HEAD -- >nul 2>&1 || (
  echo FAIL tracked files have local changes; use a clean checkout of the reviewed commit >> "%LOG%"
  echo FAIL tracked files have local changes; use a clean checkout of the reviewed commit
  goto :fail
)
call :step "Python version (3.11 expected)" %PYTHON% --version || goto :fail
call :step "Report reconciliation (read-only)" %PYTHON% -m ws3_evaluation.report_scope30 --check || goto :fail
call :step "Report and evidence tests" %PYTHON% -m pytest -p no:cacheprovider tests/test_report_scope30.py tests/test_generation_report_sections.py tests/test_report_methods_scope30.py tests/test_analysis_outputs_scope30.py || goto :fail

if /I "%MODE%"=="BUILD" (
  call :step "Install optional report build packages" %PYTHON% -m pip install -r requirements-report.txt || goto :fail
  call :step "Install Playwright Chromium" %PYTHON% -m playwright install chromium || goto :fail
  call :step "Rebuild PDF outside the repository" %PYTHON% -m ws3_evaluation.report_scope30 --build-pdf ".scratch\issue17\REPORT.rebuild.pdf" || goto :fail
  echo Rebuilt PDF: %CD%\.scratch\issue17\REPORT.rebuild.pdf
  echo Compare it page by page with docs\REPORT.pdf; byte equality is not expected across machines.
)

git diff --quiet HEAD -- >nul 2>&1 || (
  echo FAIL tracked files changed during the run >> "%LOG%"
  echo FAIL tracked files changed during the run
  goto :fail
)
echo PASS %MODE% at %HEAD% >> "%LOG%"
echo PASS %MODE% at %HEAD%
echo Log: %LOG%
exit /b 0

:step
rem %1 is the label; the remaining arguments are the command. Output goes to the log.
set "LABEL=%~1"
shift
set "CMD="
:collect
if "%~1"=="" goto :runstep
set "CMD=%CMD% %1"
shift
goto :collect
:runstep
echo.>> "%LOG%"
echo === %LABEL%: %CMD% >> "%LOG%"
echo --- %LABEL%
%CMD% >> "%LOG%" 2>&1
if errorlevel 1 (
  echo FAIL %LABEL% >> "%LOG%"
  echo FAIL %LABEL% ^(see %LOG%^)
  exit /b 1
)
echo     ok
exit /b 0

:fail
echo Stopped at the first failure. Log: %LOG%
exit /b 1
