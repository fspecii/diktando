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
    CURRENT_VERSION = "v1.1"  # This should match your current release version

    def __init__(self):
        super().__init__()
        self.executable_path = self._get_executable_path()

    def _get_executable_path(self):
        """Get the path of the current executable"""
        if getattr(sys, 'frozen', False):
            return sys.executable
        return None

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
            
            # Create a temporary file path
            temp_path = os.path.join(os.path.dirname(self.executable_path), 
                                   f"DiktandoApp_new.exe")

            # Download with progress updates
            block_size = 1024
            downloaded = 0

            with open(temp_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    progress = int((downloaded / total_size) * 100)
                    self.update_progress_signal.emit(progress)

            # Create a batch file to replace the executable
            batch_path = os.path.join(os.path.dirname(self.executable_path), 
                                    "update.bat")
            
            with open(batch_path, 'w') as f:
                f.write('@echo off\n')
                f.write('timeout /t 1 /nobreak >nul\n')  # Wait for original process to exit
                f.write(f'move /y "{temp_path}" "{self.executable_path}"\n')
                f.write(f'start "" "{self.executable_path}"\n')
                f.write(f'del "%~f0"\n')  # Delete the batch file itself

            # Start the batch file and exit the application
            subprocess.Popen([batch_path], shell=True)
            self.update_success_signal.emit()

        except Exception as e:
            self.update_error_signal.emit(f"Failed to install update: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass 