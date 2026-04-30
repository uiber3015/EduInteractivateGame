@echo off
setlocal

cd /d "%~dp0"

echo 正在启动作品展示，请稍候...
echo.

python launch_showcase.py
if errorlevel 1 goto fallback_py
goto end

:fallback_py
py -3 launch_showcase.py
if errorlevel 1 goto failed
goto end

:failed
echo.
echo 启动失败，请确认：
echo 1. 已安装 Python 3
echo 2. 已安装 requirements.txt 中的依赖
echo 3. output 目录下存在可用作品
pause
exit /b 1

:end
pause
endlocal
