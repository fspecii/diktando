import os
import sys
import json
import requests
import subprocess
from packaging import version
from PyQt5.QtCore import QObject, pyqtSignal

class UpdateChecker(QObject):
    """Handles checking for and applying updates from GitHub releases"""
    update_available_signal = pyqtSignal(str, str)  # current_version, latest_version
    update_progress_signal = pyqtSignal(int)  # progress percentage
    update_error_signal = pyqtSignal(str)  # error message
    update_success_signal = pyqtSignal()

    GITHUB_API_URL = "https://api.github.com/repos/fspecii/diktando/releases"
    CURRENT_VERSION = "v1.2"  # This should match your current release version

    def __init__(self):
        super().__init__()
        self.executable_path = self._get_executable_path()

    def _get_executable_path(self):
        """Get the path of the current executable"""
        if getattr(sys, 'frozen', False):
            return sys.executable
        else:
            # For development/non-frozen environment, use the script path
            return os.path.abspath(sys.argv[0])

    def check_for_updates(self, silent=True):
        """Check for updates on GitHub
        
        Args:
            silent (bool): If True, don't emit signals for no updates available
        """
        try:
            response = requests.get(self.GITHUB_API_URL, timeout=5)
            response.raise_for_status()
            releases = response.json()

            if not releases:
                if not silent:
                    self.update_error_signal.emit("No releases found")
                return

            latest_release = releases[0]
            latest_version = latest_release['tag_name']

            if version.parse(latest_version.lstrip('v')) > version.parse(self.CURRENT_VERSION.lstrip('v')):
                self.update_available_signal.emit(self.CURRENT_VERSION, latest_version)
                return latest_release
            elif not silent:
                self.update_error_signal.emit("You are already running the latest version")

        except requests.RequestException as e:
            if not silent:
                self.update_error_signal.emit(f"Failed to check for updates: {str(e)}")
        except Exception as e:
            if not silent:
                self.update_error_signal.emit(f"Unexpected error while checking for updates: {str(e)}")
        
        return None

    def download_and_install_update(self, release):
        """Download and install the latest release"""
        temp_path = None
        batch_path = None
        log_path = None
        
        try:
            # Find the asset that matches our platform
            asset = None
            for a in release['assets']:
                if a['name'].endswith('.exe'):  # For Windows
                    asset = a
                    break

            if not asset:
                self.update_error_signal.emit("No compatible release asset found")
                return

            # Download the new version
            response = requests.get(asset['browser_download_url'], stream=True)
            response.raise_for_status()

            # Get the total file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Create paths for temporary files
            app_dir = os.path.dirname(self.executable_path)
            temp_path = os.path.join(app_dir, "DiktandoApp_new.exe")
            batch_path = os.path.join(app_dir, "update.bat")
            log_path = os.path.join(app_dir, "update_log.txt")

            # Download with progress updates
            block_size = 1024
            downloaded = 0

            with open(temp_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    progress = int((downloaded / total_size) * 100)
                    self.update_progress_signal.emit(progress)

            # Create a batch file to handle the update process
            with open(batch_path, 'w') as f:
                f.write('@echo off\n')
                f.write('echo Starting update process... > "%s"\n' % log_path)
                f.write('timeout /t 2 /nobreak >nul\n')  # Wait longer for original process to exit
                
                # Try to terminate any remaining instances
                f.write('taskkill /F /IM "%s" >nul 2>&1\n' % os.path.basename(self.executable_path))
                f.write('timeout /t 1 /nobreak >nul\n')
                
                # Attempt to move the new executable
                f.write('echo Attempting to replace executable... >> "%s"\n' % log_path)
                f.write('if exist "%s" (\n' % temp_path)
                f.write('    move /y "%s" "%s" >> "%s" 2>&1\n' % (temp_path, self.executable_path, log_path))
                f.write('    if errorlevel 1 (\n')
                f.write('        echo Failed to move new executable >> "%s"\n' % log_path)
                f.write('        exit /b 1\n')
                f.write('    )\n')
                f.write(') else (\n')
                f.write('    echo New executable not found >> "%s"\n' % log_path)
                f.write('    exit /b 1\n')
                f.write(')\n')
                
                # Verify the new executable exists
                f.write('if exist "%s" (\n' % self.executable_path)
                f.write('    echo Update successful, launching application... >> "%s"\n' % log_path)
                f.write('    start "" "%s"\n' % self.executable_path)
                f.write(') else (\n')
                f.write('    echo Failed to verify new executable >> "%s"\n' % log_path)
                f.write('    exit /b 1\n')
                f.write(')\n')
                
                # Clean up
                f.write('echo Cleaning up... >> "%s"\n' % log_path)
                f.write('timeout /t 2 /nobreak >nul\n')
                f.write('del "%s" >nul 2>&1\n' % log_path)
                f.write('del "%~f0"\n')  # Delete the batch file itself

            # Start the batch file and emit success signal
            subprocess.Popen([batch_path], shell=True)
            self.update_success_signal.emit()

        except Exception as e:
            self.update_error_signal.emit(f"Failed to install update: {str(e)}")
            # Clean up temporary files
            for path in [temp_path, batch_path, log_path]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass 