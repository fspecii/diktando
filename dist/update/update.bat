@echo off
echo Updating Diktando...
timeout /t 2 /nobreak > nul
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"
taskkill /F /IM "DiktandoApp_v1.1.exe" 2>nul
xcopy /E /I /Y "C:\ai\diktando\dist\update\DiktandoApp_v1.exe" "C:\ai\diktando\dist\*"
if errorlevel 1 (
    echo Update failed. Please try again.
    pause
    exit /b 1
)
echo Update successful!
start "" "C:\ai\diktando\dist\DiktandoApp_v1.1.exe"
cd ..
rd /s /q update
exit
