@echo off
chcp 936 >nul
setlocal EnableDelayedExpansion

REM ï¿½ï¿½ï¿½ï¿½ PYTHONUTF8=1 ï¿½ï¿½ï¿½ï¿½condaï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
set "PYTHONUTF8=1"

REM ï¿½Þ¸ï¿½ï¿½ï¿½ï¿½ï¿½Ô±Ä£Ê½ï¿½ï¿½%CD%ï¿½ï¿½ï¿½system32ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
REM Ê¹ï¿½Ã½Å±ï¿½ï¿½ï¿½ï¿½ï¿½Ä¿Â¼ï¿½ï¿½Îªï¿½ï¿½ï¿½ï¿½Ä¿Â¼
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM ï¿½ï¿½ï¿½condaï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Â·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
set CONDA_ENV_NAME=manga-12
set CONDA_ENV_PATH=%SCRIPT_DIR%\conda_env
set MINICONDA_ROOT=%SCRIPT_DIR%\Miniconda3

REM ï¿½ï¿½ï¿½Â·ï¿½ï¿½ï¿½Ç·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ASCIIï¿½Ö·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ÄµÈ£ï¿½
REM Ê¹ï¿½ï¿½PowerShellï¿½ï¿½ï¿½Ð¸ï¿½ï¿½É¿ï¿½ï¿½Ä¼ï¿½ï¿?
set "TEMP_CHECK_PATH=%SCRIPT_DIR%"
powershell -Command "$path = '%TEMP_CHECK_PATH%'; if ($path -match '[^\x00-\x7F]') { exit 1 } else { exit 0 }" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    REM Â·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Ä£ï¿½Ê¹ï¿½Ã´ï¿½ï¿½Ì¸ï¿½Ä¿Â¼ï¿½ï¿½Miniconda
    set MINICONDA_ROOT=%~d0\Miniconda3
)

REM ï¿½È¼ï¿½ï¿½ÏµÍ³conda
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 goto :check_local_conda_s2

REM ï¿½ï¿½âµ½ÏµÍ³condaï¿½ï¿½ï¿½ï¿½È¡Êµï¿½ï¿½Â·ï¿½ï¿½
REM ï¿½ï¿½ï¿½ï¿½1: ï¿½ï¿½CONDA_EXEï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È¡ï¿½ï¿½ï¿½ï¿½É¿ï¿½ï¿½ï¿?
if defined CONDA_EXE (
    for %%p in ("%CONDA_EXE%\..\..") do set "MINICONDA_ROOT=%%~fp"
)

REM ï¿½ï¿½ï¿½ï¿½2: ï¿½ï¿½CONDA_PREFIXï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½È¡
if "!MINICONDA_ROOT!"=="" (
    if defined CONDA_PREFIX (
        set "MINICONDA_ROOT=%CONDA_PREFIX%"
    )
)

REM ï¿½ï¿½ï¿½ï¿½3: Ê¹ï¿½ï¿½ conda info --base
if "!MINICONDA_ROOT!"=="" (
    for /f "delims=" %%i in ('conda info --base 2^>nul') do (
        set "TEMP_PATH=%%i"
        if exist "!TEMP_PATH!\Scripts\conda.exe" (
            set "MINICONDA_ROOT=%%i"
        )
    )
)

REM ï¿½ï¿½ï¿½ï¿½4: ï¿½ï¿½ where conda ï¿½ï¿½ï¿½ï¿½Â·ï¿½ï¿½
if "!MINICONDA_ROOT!"=="" (
    for /f "delims=" %%i in ('where conda 2^>nul') do (
        if "!MINICONDA_ROOT!"=="" (
            if "%%~xi"==".exe" (
                for %%p in ("%%~dpi..") do set "MINICONDA_ROOT=%%~fp"
            ) else if "%%~xi"==".bat" (
                for %%p in ("%%~dpi..\..") do set "MINICONDA_ROOT=%%~fp"
            )
        )
    )
)

goto :check_env_s2

:check_local_conda_s2
REM ï¿½ï¿½é±¾ï¿½ï¿½Minicondaï¿½ï¿½ï¿½ï¿½ï¿½È½Å±ï¿½Ä¿Â¼ï¿½ï¿½
if exist "%SCRIPT_DIR%\Miniconda3\Scripts\conda.exe" (
    set MINICONDA_ROOT=%SCRIPT_DIR%\Miniconda3
    echo [INFO] ¼ì²âµ½±¾µØ Miniconda: %MINICONDA_ROOT%
    call "%MINICONDA_ROOT%\Scripts\activate.bat"
    goto :check_env_s2
)

REM ï¿½ï¿½ï¿½ï¿½ï¿½Ì¸ï¿½Ä¿Â¼
if exist "%~d0\Miniconda3\Scripts\conda.exe" (
    set MINICONDA_ROOT=%~d0\Miniconda3
    echo [INFO] ¼ì²âµ½±¾µØ Miniconda: %MINICONDA_ROOT%
    call "%MINICONDA_ROOT%\Scripts\activate.bat"
    goto :check_env_s2
)

echo [ERROR] Î´¼ì²âµ½ Conda
echo ÇëÔËÐÐ ²½Öè1-Ê×´Î°²×°.bat °²×° Miniconda
pause
exit /b 1

:check_env_s2

REM ï¿½ï¿½é»·ï¿½ï¿½ï¿½Ç·ï¿½ï¿½ï¿½Ú£ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
REM Ê¹ï¿½ï¿½ /B Ñ¡ï¿½ï¿½ï¿½ï¿½Ð¾ï¿½È·Æ¥ï¿½ï¿½ï¿½ï¿½ï¿½×£ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½Æ¥ï¿½ï¿½Â·ï¿½ï¿½ï¿½Ðµï¿½ï¿½Ä±ï¿?
call conda info --envs 2>nul | findstr /B /C:"%CONDA_ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [INFO] ¼ì²âµ½ÃüÃû»·¾³: %CONDA_ENV_NAME%
    goto :env_check_ok
)

REM ï¿½ï¿½ï¿½É°æ±¾Â·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
if exist "%CONDA_ENV_PATH%\python.exe" (
    echo [INFO] ¼ì²âµ½Â·¾¶Ö¸¶¨µÄ¾É°æ±¾»·¾³
    goto :env_check_ok
)

REM Ã»ï¿½ï¿½ï¿½ÎºÎ»ï¿½ï¿½ï¿½
echo [ERROR] Î´¼ì²âµ½ Conda »·¾³
echo ÇëÔËÐÐ ²½Öè1-Ê×´Î°²×°.bat ´´½¨»·¾³
pause
exit /b 1

:env_check_ok

REM ï¿½ï¿½È·ï¿½ï¿½ conda ï¿½Ñ³ï¿½Ê¼ï¿½ï¿½
if not exist "%MINICONDA_ROOT%\Scripts\activate.bat" goto :try_activate_s2
call "%MINICONDA_ROOT%\Scripts\activate.bat"

:try_activate_s2
REM ï¿½ï¿½ï¿½ï¿½1: conda activate ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
call conda activate "%CONDA_ENV_NAME%" 2>nul && goto :activated_ok_s2

REM ï¿½ï¿½ï¿½ï¿½2: activate.bat ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
echo [INFO] ³¢ÊÔ±¸ÓÃ¼¤»î·½Ê½...
if not exist "%MINICONDA_ROOT%\Scripts\activate.bat" goto :try_manual_path_s2
call "%MINICONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV_NAME%" 2>nul && goto :activated_ok_s2

:try_manual_path_s2
REM ï¿½ï¿½ï¿½ï¿½3: ï¿½ï¿½È¡ï¿½ï¿½ï¿½ï¿½Â·ï¿½ï¿½ï¿½ï¿½ï¿½Ö¶ï¿½ï¿½ï¿½ï¿½ï¿½PATH
for /f "tokens=2" %%i in ('conda info --envs 2^>nul ^| findstr /B /C:"%CONDA_ENV_NAME%"') do set "ENV_PATH=%%i"
if not defined ENV_PATH goto :try_legacy_env_s2
if not exist "!ENV_PATH!\python.exe" goto :try_legacy_env_s2
echo [INFO] Ê¹ÓÃÊÖ¶¯ PATH ¼¤»î·½Ê½...
set "PATH=!ENV_PATH!;!ENV_PATH!\Library\mingw-w64\bin;!ENV_PATH!\Library\usr\bin;!ENV_PATH!\Library\bin;!ENV_PATH!\Scripts;!ENV_PATH!\bin;%PATH%"
set "CONDA_PREFIX=!ENV_PATH!"
set "CONDA_DEFAULT_ENV=%CONDA_ENV_NAME%"
echo [INFO] ÒÑ¼¤»î»·¾³: %CONDA_ENV_NAME%
goto :activated_ok_s2

:try_legacy_env_s2
REM ï¿½ï¿½ï¿½ï¿½4: ï¿½É°æ±¾Â·ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
if not exist "%CONDA_ENV_PATH%\python.exe" goto :activate_failed_s2
echo [INFO] Ê¹ÓÃÂ·¾¶Ö¸¶¨µÄ¾É°æ±¾»·¾³...
echo [INFO] Ê¹ÓÃÊÖ¶¯ PATH ¼¤»î·½Ê½...
set "PATH=%CONDA_ENV_PATH%;%CONDA_ENV_PATH%\Library\mingw-w64\bin;%CONDA_ENV_PATH%\Library\usr\bin;%CONDA_ENV_PATH%\Library\bin;%CONDA_ENV_PATH%\Scripts;%CONDA_ENV_PATH%\bin;%PATH%"
set "CONDA_PREFIX=%CONDA_ENV_PATH%"
set "CONDA_DEFAULT_ENV=%CONDA_ENV_PATH%"
goto :activated_ok_s2

:activate_failed_s2
echo [ERROR] ÎÞ·¨¼¤»î»·¾³
echo Çë³¢ÊÔ: ´ò¿ªÃüÁîÌáÊ¾·ûÔËÐÐ conda init cmd.exe£¬È»ºóÖØÊÔ
pause
exit /b 1

:activated_ok_s2

REM ï¿½ï¿½ï¿½ï¿½Ç·ï¿½ï¿½Ð±ï¿½Ð¯ï¿½ï¿?Git
if not exist "PortableGit\cmd\git.exe" goto :skip_git_s2
set "PATH=%SCRIPT_DIR%\PortableGit\cmd;%PATH%"
:skip_git_s2

REM ï¿½Ð»ï¿½ï¿½ï¿½ï¿½ï¿½Ä¿ï¿½ï¿½Ä¿Â¼(È·ï¿½ï¿½Pythonï¿½ï¿½ï¿½ï¿½È·ï¿½Òµï¿½Ä£ï¿½ï¿½)
cd /d "%~dp0"

REM Ö±ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ Qt ï¿½ï¿½ï¿½ï¿½
python desktop_qt_ui\main.py
pause
