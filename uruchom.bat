@echo off
setlocal
cd /d "%~dp0"

call :find_python
if not defined PYTHON_EXE (
  where winget.exe >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Brakuje Pythona i programu Windows App Installer - winget.
    echo Zainstaluj Python 3.13 64-bit z https://www.python.org/downloads/windows/
    echo, a nastepnie uruchom ten plik ponownie.
    echo.
    pause
    exit /b 1
  )
  echo.
  echo Pierwsze uruchomienie: instalowanie Pythona 3.13...
  winget install -e --id Python.Python.3.13 --accept-package-agreements --accept-source-agreements
  if errorlevel 1 (
    echo.
    echo Nie udalo sie automatycznie zainstalowac Pythona.
    echo Zainstaluj go z https://www.python.org/downloads/windows/ i uruchom plik ponownie.
    pause
    exit /b 1
  )
  call :find_python
)

if not defined PYTHON_EXE (
  echo.
  echo Python zostal zainstalowany, ale system jeszcze go nie wykryl.
  echo Zamknij to okno i uruchom ponownie uruchom.bat.
  pause
  exit /b 1
)

rem A copied virtual environment contains the absolute path to the Python
rem installation from the original computer. Verify it before using it.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Wykryto srodowisko Pythona z innego komputera. Tworzenie nowego...
    rmdir /s /q ".venv"
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Pierwsze uruchomienie: pobieranie bibliotek transkrypcji...
  "%PYTHON_EXE%" %PYTHON_ARGS% -m venv .venv
  if errorlevel 1 goto :venv_error
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :venv_error
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :venv_error
)

rem Make CUDA libraries installed with pip visible to faster-whisper.
if exist ".venv\Lib\site-packages\nvidia" (
  for /d %%D in (".venv\Lib\site-packages\nvidia\*") do (
    if exist "%%~fD\bin" set "PATH=%%~fD\bin;%PATH%"
  )
)

".venv\Scripts\python.exe" app.py
exit /b %errorlevel%

:find_python
set "PYTHON_EXE="
set "PYTHON_ARGS="
where py.exe >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_EXE=py.exe"
  set "PYTHON_ARGS=-3"
  exit /b 0
)
where python.exe >nul 2>nul
if not errorlevel 1 (
  set "PYTHON_EXE=python.exe"
  exit /b 0
)
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do (
  if exist "%%~fD\python.exe" set "PYTHON_EXE=%%~fD\python.exe"
)
exit /b 0

:venv_error
echo.
echo Nie udalo sie pobrac wymaganych bibliotek. Sprawdz polaczenie z internetem i uruchom plik ponownie.
pause
exit /b 1
