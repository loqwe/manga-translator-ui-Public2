@echo off
chcp 936 >nul
setlocal EnableDelayedExpansion

REM 设置 PYTHONUTF8=1 避免conda编码错误
set "PYTHONUTF8=1"

REM 修复管理员模式下%CD%变成system32的问题
REM 使用脚本所在目录作为工作目录
cd /d "%~dp0"
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

REM 检查conda环境（兼容命名环境和路径环境）
set CONDA_ENV_NAME=manga-env
set CONDA_ENV_PATH=%SCRIPT_DIR%\conda_env
set MINICONDA_ROOT=%SCRIPT_DIR%\Miniconda3

REM 检测路径是否包含非ASCII字符（中文等）
REM 使用PowerShell进行更可靠的检测
set "TEMP_CHECK_PATH=%SCRIPT_DIR%"
powershell -Command "$path = '%TEMP_CHECK_PATH%'; if ($path -match '[^\x00-\x7F]') { exit 1 } else { exit 0 }" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    REM 路径包含中文，使用磁盘根目录的Miniconda
    set MINICONDA_ROOT=%~d0\Miniconda3
)

REM 先检查系统conda
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 goto :check_local_conda_s3

REM 检测到系统conda，获取实际路径
REM 方法1: 从CONDA_EXE环境变量获取（最可靠）
if defined CONDA_EXE (
    for %%p in ("%CONDA_EXE%\..\..") do set "MINICONDA_ROOT=%%~fp"
)

REM 方法2: 从CONDA_PREFIX环境变量获取
if "!MINICONDA_ROOT!"=="" (
    if defined CONDA_PREFIX (
        set "MINICONDA_ROOT=%CONDA_PREFIX%"
    )
)

REM 方法3: 使用 conda info --base
if "!MINICONDA_ROOT!"=="" (
    for /f "delims=" %%i in ('conda info --base 2^>nul') do (
        set "TEMP_PATH=%%i"
        if exist "!TEMP_PATH!\Scripts\conda.exe" (
            set "MINICONDA_ROOT=%%i"
        )
    )
)

REM 方法4: 从 where conda 解析路径
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

goto :check_env_s3

:check_local_conda_s3
REM 检查本地Miniconda（优先脚本目录）
if exist "%SCRIPT_DIR%\Miniconda3\Scripts\conda.exe" (
    set MINICONDA_ROOT=%SCRIPT_DIR%\Miniconda3
    echo [INFO] 检测到本地 Miniconda: %MINICONDA_ROOT%
    call "%MINICONDA_ROOT%\Scripts\activate.bat"
    goto :check_env_s3
)

REM 检查磁盘根目录
if exist "%~d0\Miniconda3\Scripts\conda.exe" (
    set MINICONDA_ROOT=%~d0\Miniconda3
    echo [INFO] 检测到本地 Miniconda: %MINICONDA_ROOT%
    call "%MINICONDA_ROOT%\Scripts\activate.bat"
    goto :check_env_s3
)

echo [ERROR] 未检测到 Conda
echo 请先运行 步骤1-首次安装.bat 安装 Miniconda
pause
exit /b 1

:check_env_s3

REM 检查环境是否存在（优先命名环境）
REM 使用 /B 选项进行精确匹配行首，避免误匹配路径中的文本
call conda info --envs 2>nul | findstr /B /C:"%CONDA_ENV_NAME%" >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo [INFO] 检测到命名环境: %CONDA_ENV_NAME%
    goto :env_check_ok
)

REM 检查旧版本路径环境
if exist "%CONDA_ENV_PATH%\python.exe" (
    echo [INFO] 检测到路径环境（旧版本）
    goto :env_check_ok
)

REM 没有任何环境
echo [ERROR] 未检测到Conda环境
echo 请先运行 步骤1-首次安装.bat 创建环境
pause
exit /b 1

:env_check_ok

REM 先确保 conda 已初始化
if not exist "%MINICONDA_ROOT%\Scripts\activate.bat" goto :try_activate_s3
call "%MINICONDA_ROOT%\Scripts\activate.bat"

:try_activate_s3
REM 方法1: conda activate 命名环境
call conda activate "%CONDA_ENV_NAME%" 2>nul && goto :activated_ok_s3

REM 方法2: activate.bat 激活命名环境
echo [INFO] 尝试备用激活方式...
if not exist "%MINICONDA_ROOT%\Scripts\activate.bat" goto :try_manual_path_s3
call "%MINICONDA_ROOT%\Scripts\activate.bat" "%CONDA_ENV_NAME%" 2>nul && goto :activated_ok_s3

:try_manual_path_s3
REM 方法3: 获取环境路径并手动设置PATH
for /f "tokens=2" %%i in ('conda info --envs 2^>nul ^| findstr /B /C:"%CONDA_ENV_NAME%"') do set "ENV_PATH=%%i"
if not defined ENV_PATH goto :try_legacy_env_s3
if not exist "!ENV_PATH!\python.exe" goto :try_legacy_env_s3
echo [INFO] 使用手动PATH激活方式...
set "PATH=!ENV_PATH!;!ENV_PATH!\Library\mingw-w64\bin;!ENV_PATH!\Library\usr\bin;!ENV_PATH!\Library\bin;!ENV_PATH!\Scripts;!ENV_PATH!\bin;%PATH%"
set "CONDA_PREFIX=!ENV_PATH!"
set "CONDA_DEFAULT_ENV=%CONDA_ENV_NAME%"
echo [INFO] 已激活环境: %CONDA_ENV_NAME%
goto :activated_ok_s3

:try_legacy_env_s3
REM 方法4: 旧版本路径环境
if not exist "%CONDA_ENV_PATH%\python.exe" goto :activate_failed_s3
echo [INFO] 激活路径环境（旧版本）...
echo [INFO] 使用手动PATH激活方式...
set "PATH=%CONDA_ENV_PATH%;%CONDA_ENV_PATH%\Library\mingw-w64\bin;%CONDA_ENV_PATH%\Library\usr\bin;%CONDA_ENV_PATH%\Library\bin;%CONDA_ENV_PATH%\Scripts;%CONDA_ENV_PATH%\bin;%PATH%"
set "CONDA_PREFIX=%CONDA_ENV_PATH%"
set "CONDA_DEFAULT_ENV=%CONDA_ENV_PATH%"
goto :activated_ok_s3

:activate_failed_s3
echo [ERROR] 无法激活环境
echo 请尝试: 打开新命令提示符，运行 conda init cmd.exe，然后重试
pause
exit /b 1

:activated_ok_s3

REM 检查是否有便携版 Git
if not exist "PortableGit\cmd\git.exe" goto :check_system_git_s3
set "GIT=%SCRIPT_DIR%\PortableGit\cmd\git.exe"
set "PATH=%SCRIPT_DIR%\PortableGit\cmd;%PATH%"
goto :git_done_s3

:check_system_git_s3
git --version >nul 2>&1 && set GIT=git && goto :git_done_s3
REM Git不可用，跳过版本检查
goto :skip_version_check

:git_done_s3

REM 使用Python脚本快速检查版本（避免批处理冒号问题）
python packaging\check_version.py --brief 2>nul

:skip_version_check
REM 切换到项目根目录
cd /d "%~dp0"

REM 启动 Qt 界面
echo 正在启动...
echo ========================================
echo.
python desktop_qt_ui\main.py
pause
