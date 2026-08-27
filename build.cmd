@echo off
rem ---------------------------------------------------------------------------
rem Windows 一键配置 + 构建(提示信息用英文写:cmd 按 GBK 读脚本,
rem 中文 echo 会乱码;注释不参与输出,不受影响)。
rem   build.cmd                      配置并构建全部 sample(Release)
rem   build.cmd Debug                指定构建类型
rem   build.cmd Release reduction    只构建某个 target
rem 脚本自己找齐 MSVC / CMake / Ninja / nvcc,不依赖当前终端的 PATH。
rem ---------------------------------------------------------------------------
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BUILD_TYPE=%~1"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"
set "TARGET=%~2"

rem ---- 1. MSVC:nvcc 需要 cl.exe 作为 host 编译器,靠 vcvars64 准备好环境 ----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Install Visual Studio with the C++ workload.
    exit /b 1
)
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo [ERROR] No Visual Studio installation with MSVC tools found.
    exit /b 1
)
call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem ---- 2. CMake / Ninja:PATH 里没有就退回 Visual Studio 自带的那份 ----
where /q cmake || set "PATH=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
where /q ninja || set "PATH=%VSPATH%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
where /q cmake || (echo [ERROR] cmake not found. & exit /b 1)
where /q ninja || (echo [ERROR] ninja not found. & exit /b 1)

rem ---- 3. nvcc:优先 CUDA_PATH,再兜底扫默认安装位置 ----
if not defined CUDA_PATH (
    for %%d in ("%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA") do (
        for /f "delims=" %%v in ('dir /b /o-n "%%~d\v*" 2^>nul') do (
            if not defined CUDA_PATH if exist "%%~d\%%v\bin\nvcc.exe" set "CUDA_PATH=%%~d\%%v"
        )
    )
)
if defined CUDA_PATH set "PATH=%CUDA_PATH%\bin;%PATH%"
where /q nvcc || (
    echo [ERROR] nvcc not found. Install the CUDA Toolkit and point CUDA_PATH at it,
    echo         e.g.  setx CUDA_PATH "F:\software\NVIDIA"
    exit /b 1
)

rem ---- 4. 配置 + 构建 ----
cmake -S "%ROOT%" -B "%ROOT%\build" -G Ninja -DCMAKE_BUILD_TYPE=%BUILD_TYPE% || exit /b 1
if "%TARGET%"=="" (
    cmake --build "%ROOT%\build" || exit /b 1
) else (
    cmake --build "%ROOT%\build" --target %TARGET% || exit /b 1
)

echo.
echo [OK] Executables are in %ROOT%\bin
endlocal
