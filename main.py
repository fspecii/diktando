#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import time
import wave
import shutil
import platform
import threading
import subprocess
import numpy as np
import sounddevice as sd
import soundfile as sf
import logging
from pathlib import Path
from datetime import datetime
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QUrl, QMimeData, QPoint, QRect
from PyQt5.QtGui import QIcon, QTextCursor, QPainter, QColor, QFont, QDesktopServices, QKeySequence, QPalette, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QProgressBar, QStatusBar,
    QFileDialog, QMessageBox, QComboBox, QCheckBox, QGroupBox,
    QRadioButton, QTabWidget, QSpinBox, QLineEdit, QSystemTrayIcon,
    QMenu, QAction, QDialog, QDialogButtonBox, QKeySequenceEdit, QButtonGroup
)

# Global debug flag - set to 1 to enable logging, 0 to disable
DEBUG = 1

# Initialize logger variable at the top level to avoid circular dependencies
logger = logging.getLogger('diktando')

# Setup logging
def setup_logging(debug_mode=False):
    """Set up logging based on debug mode"""
    global logger
    
    try:
        if debug_mode:
            # Determine the application directory - handle both script and frozen app
            if getattr(sys, 'frozen', False):
                # Running as compiled executable
                app_dir = os.path.dirname(sys.executable)
            else:
                # Running as script
                app_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Create logs directory if it doesn't exist
            logs_dir = os.path.join(app_dir, 'logs')
            try:
                os.makedirs(logs_dir, exist_ok=True)
                original_print(f"Created logs directory at: {logs_dir}")
            except Exception as e:
                original_print(f"Error creating logs directory: {str(e)}")
                # Fallback to temp directory if we can't create logs in app dir
                import tempfile
                logs_dir = os.path.join(tempfile.gettempdir(), 'diktando_logs')
                os.makedirs(logs_dir, exist_ok=True)
                original_print(f"Using fallback logs directory: {logs_dir}")
            
            # Create log file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(logs_dir, f"diktando_{timestamp}.log")
            
            try:
                # Set up file handler
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                
                # Set up console handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.DEBUG)
                console_formatter = logging.Formatter('%(levelname)s: %(message)s')
                console_handler.setFormatter(console_formatter)
                
                # Configure root logger
                root_logger = logging.getLogger()
                root_logger.setLevel(logging.DEBUG)
                root_logger.addHandler(file_handler)
                root_logger.addHandler(console_handler)
                
                # Configure our application logger
                logger.setLevel(logging.DEBUG)
                original_print(f"Logging initialized. Log file: {log_file}")
                
                return log_file
            except Exception as e:
                original_print(f"Error setting up logging: {str(e)}")
                # Create a dummy logger that does nothing
                logger.setLevel(logging.CRITICAL)
        else:
            # Disable logging
            logging.getLogger().setLevel(logging.CRITICAL)
            logging.disable(logging.CRITICAL)
            # Configure our application logger
            logger.setLevel(logging.CRITICAL)
    except Exception as e:
        original_print(f"Critical error in setup_logging: {str(e)}")
        # Ensure we always have a logger object even if setup fails
        logger.setLevel(logging.CRITICAL)
    
    return None

# Custom print function that also logs
original_print = print
def custom_print(*args, **kwargs):
    """Custom print function that also logs to file"""
    # Call the original print function
    original_print(*args, **kwargs)
    
    # Also log the message if debug mode is enabled
    if DEBUG:
        try:
            message = " ".join(str(arg) for arg in args)
            # Only use logger if it's properly initialized
            if logger and hasattr(logger, 'info'):
                logger.info(message)
        except Exception as e:
            # If logging fails, just print the error but don't crash
            original_print(f"Logging error: {str(e)}")

# Replace the built-in print function with our custom one
print = custom_print

# Import custom modules
from clipboard_manager import ClipboardManager
from updater import UpdateChecker
from llm_processor import LLMProcessor
from llm_settings_dialog import LLMSettingsDialog

# Import keyboard hook library based on platform
if platform.system() == "Windows":
    import keyboard
    # Add CREATE_NO_WINDOW flag for Windows
    SUBPROCESS_FLAGS = 0x08000000  # CREATE_NO_WINDOW
elif platform.system() == "Darwin":  # macOS
    try:
        from pynput import keyboard
    except ImportError:
        print("Please install pynput: pip install pynput")
    SUBPROCESS_FLAGS = 0
else:  # Linux
    try:
        from pynput import keyboard
    except ImportError:
        print("Please install pynput: pip install pynput")
    SUBPROCESS_FLAGS = 0

class AudioRecorder(QThread):
    """Thread for recording audio"""
    update_signal = pyqtSignal(np.ndarray)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, sample_rate=16000, channels=1, device=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.recording = False
        self.audio_data = []
        self.min_volume_threshold = 0.001  # Minimum volume threshold
        self.gain = 5.0  # Audio gain multiplier
        
    def run(self):
        try:
            print(f"Starting audio recording with device: {self.device}, sample rate: {self.sample_rate}, channels: {self.channels}")
            
            # Configure input stream with higher buffer size and blocksize
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                callback=self.audio_callback,
                blocksize=2048,  # Larger block size for better quality
                latency='low'    # Low latency for better responsiveness
            ) as stream:
                print("Audio stream opened successfully")
                while self.recording:
                    time.sleep(0.1)
            
            print(f"Recording stopped, collected {len(self.audio_data)} audio chunks")
            
            if not self.audio_data:
                self.error_signal.emit("No audio data recorded")
                return
                
            # Convert to numpy array
            audio_array = np.concatenate(self.audio_data, axis=0)
            print(f"Audio array shape: {audio_array.shape}, min: {audio_array.min()}, max: {audio_array.max()}")
            
            # Apply gain and normalization
            audio_array = self.process_audio(audio_array)
            
            # Save to WAV file
            temp_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(temp_dir, "recording.wav")
            
            print(f"Saving audio to {output_file}")
            sf.write(output_file, audio_array, self.sample_rate)
            
            # Verify the file was created
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                print(f"Audio file saved successfully: {file_size} bytes")
                if file_size < 1000:  # Less than 1KB
                    self.error_signal.emit(f"Warning: Audio file is very small ({file_size} bytes)")
            else:
                self.error_signal.emit("Failed to save audio file")
                return
            
            self.finished_signal.emit(output_file)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Recording error: {str(e)}\n{error_details}")
            self.error_signal.emit(f"Recording error: {str(e)}")
    
    def process_audio(self, audio_array):
        """Process the audio array to improve quality"""
        try:
            # Apply gain
            audio_array = audio_array * self.gain
            
            # Calculate RMS value
            rms = np.sqrt(np.mean(np.square(audio_array)))
            print(f"Audio RMS before normalization: {rms}")
            
            # Only normalize if the audio is too quiet
            if rms < self.min_volume_threshold:
                print(f"Audio signal is very quiet (RMS: {rms}), normalizing")
                max_amplitude = np.max(np.abs(audio_array))
                if max_amplitude > 0:
                    # Normalize to 80% of maximum possible amplitude
                    audio_array = audio_array / max_amplitude * 0.8
                    print(f"Normalized audio - New RMS: {np.sqrt(np.mean(np.square(audio_array)))}")
            
            # Ensure we don't have any clipping
            audio_array = np.clip(audio_array, -1.0, 1.0)
            
            return audio_array
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return audio_array
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        # Calculate current volume level
        volume_level = np.max(np.abs(indata))
        
        # Only emit warning if volume is very low
        if volume_level < self.min_volume_threshold:
            print("Warning: Very low audio level detected")
            
        # Apply gain in real-time for level meter
        indata_gained = indata * self.gain
        
        self.audio_data.append(indata.copy())
        self.update_signal.emit(indata_gained)
    
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.start()
    
    def stop_recording(self):
        self.recording = False


class ModelDownloader(QThread):
    """Thread for downloading models"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, model_name, model_url, output_dir):
        super().__init__()
        self.model_name = model_name
        self.model_url = model_url
        self.output_dir = output_dir
        self.is_cancelled = False
        self.temp_file = None
        
    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            output_file = os.path.join(self.output_dir, f"ggml-{self.model_name}.bin")
            self.temp_file = output_file + ".download"
            
            # Check if file already exists
            if os.path.exists(output_file):
                # File exists, check if it's complete
                try:
                    # Try to get the expected file size from the server
                    head_response = requests.head(self.model_url, timeout=10)
                    expected_size = int(head_response.headers.get('content-length', 0))
                    actual_size = os.path.getsize(output_file)
                    
                    if actual_size >= expected_size and expected_size > 0:
                        # File is already complete
                        print(f"Model file already exists and appears complete: {output_file}")
                        self.progress_signal.emit(100)
                        self.finished_signal.emit(output_file)
                        return
                    else:
                        # File exists but is incomplete, remove it and download again
                        print(f"Incomplete model file found, redownloading: {output_file}")
                        os.remove(output_file)
                except Exception as e:
                    print(f"Error checking existing file: {str(e)}, redownloading")
                    # If there's any error checking the file, just redownload it
                    if os.path.exists(output_file):
                        os.remove(output_file)
            
            # Download the file with timeout and retry logic
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries and not self.is_cancelled:
                try:
                    # Download with timeout
                    response = requests.get(self.model_url, stream=True, timeout=30)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    total_size = int(response.headers.get('content-length', 0))
                    if total_size == 0:
                        self.error_signal.emit("Error: Could not determine file size from server")
                        return
                    
                    # Create a temporary file first
                    downloaded_size = 0
                    
                    with open(self.temp_file, 'wb') as f:
                        for data in response.iter_content(chunk_size=1024):
                            if self.is_cancelled:
                                print("Download cancelled, breaking download loop")
                                break
                            
                            if data:
                                f.write(data)
                                downloaded_size += len(data)
                                progress = int(downloaded_size / total_size * 100)
                                self.progress_signal.emit(progress)
                    
                    if self.is_cancelled:
                        print("Download cancelled after download loop")
                        # Clean up the temporary file
                        self._cleanup_temp_file()
                        return
                    
                    # Verify the download
                    if os.path.exists(self.temp_file) and os.path.getsize(self.temp_file) >= total_size * 0.99:  # Allow for slight size differences
                        # Rename the temporary file to the final filename
                        if os.path.exists(output_file):
                            os.remove(output_file)
                        os.rename(self.temp_file, output_file)
                        self.temp_file = None  # Reset temp file reference after successful rename
                        self.progress_signal.emit(100)
                        self.finished_signal.emit(output_file)
                        return
                    else:
                        # Download was incomplete
                        self._cleanup_temp_file()
                        raise Exception("Download was incomplete")
                    
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        self.error_signal.emit(f"Download failed after {max_retries} attempts: {str(e)}")
                        return
                    
                    # Wait before retrying
                    print(f"Download attempt {retry_count} failed: {str(e)}. Retrying in 2 seconds...")
                    time.sleep(2)
                
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        self.error_signal.emit(f"Download error: {str(e)}")
                        return
                    
                    # Wait before retrying
                    print(f"Error during download attempt {retry_count}: {str(e)}. Retrying in 2 seconds...")
                    time.sleep(2)
            
            if self.is_cancelled:
                self.error_signal.emit("Download was cancelled")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_signal.emit(f"Download error: {str(e)}\n{error_details}")
        finally:
            # Always clean up temporary file if it exists
            self._cleanup_temp_file()
    
    def _cleanup_temp_file(self):
        """Clean up the temporary download file if it exists"""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                print(f"Cleaning up temporary file: {self.temp_file}")
                os.remove(self.temp_file)
                self.temp_file = None
            except Exception as e:
                print(f"Error cleaning up temporary file: {str(e)}")
    
    def cancel(self):
        """Cancel the download"""
        print("Download cancellation requested")
        self.is_cancelled = True


class Transcriber(QThread):
    """Thread for transcribing audio"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, audio_file, model_path, language):
        super().__init__()
        self.audio_file = audio_file
        self.model_path = model_path
        self.language = language
        
    def run(self):
        try:
            # Get user's app data directory for binaries
            app_data_dir = os.path.join(os.getenv('APPDATA') if platform.system() == "Windows" 
                                      else os.path.expanduser('~/.config'), 'Diktando')
            bin_dir = os.path.join(app_data_dir, 'bin')
            whisper_exe = os.path.join(bin_dir, "whisper-cpp.exe") if platform.system() == "Windows" else "./whisper-cpp"
            
            if not os.path.exists(whisper_exe):
                self.error_signal.emit(f"Whisper executable not found: {whisper_exe}")
                return
            
            # Check if the model file exists
            if not os.path.exists(self.model_path):
                self.error_signal.emit(f"Model file not found: {self.model_path}")
                return
            
            # Check if the audio file exists
            if not os.path.exists(self.audio_file):
                self.error_signal.emit(f"Audio file not found: {self.audio_file}")
                return
            
            # Check audio file size
            audio_size = os.path.getsize(self.audio_file)
            if audio_size == 0:
                self.error_signal.emit(f"Audio file is empty: {self.audio_file}")
                return
            elif audio_size < 1000:  # Less than 1KB
                self.error_signal.emit(f"Audio file is too small ({audio_size} bytes): {self.audio_file}")
                return
                
            # Print debug info
            print(f"Transcribing file: {self.audio_file}")
            print(f"Model path: {self.model_path}")
            print(f"Language: {self.language}")
            print(f"Audio file size: {audio_size} bytes")
            
            # Run whisper-cpp
            cmd = [
                whisper_exe,
                "-m", self.model_path,
                "-f", self.audio_file,
                "-l", self.language
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            self.progress_signal.emit(10)  # Starting
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                creationflags=SUBPROCESS_FLAGS
            )
            
            self.progress_signal.emit(50)  # Processing
            
            stdout, stderr = process.communicate()
            
            print(f"Process return code: {process.returncode}")
            print(f"Stdout: {stdout[:200]}...")  # Print first 200 chars
            print(f"Stderr: {stderr}")
            
            if process.returncode != 0:
                self.error_signal.emit(f"Transcription failed: {stderr}")
                return
            
            if not stdout.strip():
                self.error_signal.emit("Transcription produced no output")
                return
                
            self.progress_signal.emit(100)  # Finished
            self.finished_signal.emit(stdout)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_signal.emit(f"Transcription error: {str(e)}\n\n{error_details}")


class HotkeyManager(QThread):
    """Manages global hotkeys and clipboard operations"""
    hotkey_toggle_signal = pyqtSignal(str)  # For toggle mode, emits hotkey_id
    hotkey_press_signal = pyqtSignal(str)   # For push-to-talk press, emits hotkey_id
    hotkey_release_signal = pyqtSignal(str)  # For push-to-talk release, emits hotkey_id
    error_signal = pyqtSignal(str)
    paste_complete_signal = pyqtSignal(bool)  # True if successful, False otherwise
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.hotkeys = {}  # Dictionary to store multiple hotkeys: {hotkey_string: {'id': id, 'is_push_to_talk': bool, 'is_pressed': bool}}
        self.transcription = None
        self.clipboard_manager = ClipboardManager()  # Use ClipboardManager
        self.key_check_timer = None  # Timer for fallback key release detection
        self.last_press_time = 0  # To prevent multiple rapid presses
        self.debounce_interval = 0.1  # 100ms debounce
    
    def _check_hotkey_event(self, event, is_press):
        """Check if the event matches any of our hotkey combinations"""
        try:
            # Get the name of the pressed key
            if hasattr(event, 'name'):
                key_name = event.name
            else:
                key_name = str(event.scan_code)
            
            # Convert to lowercase for case-insensitive comparison
            key_name = key_name.lower()
            
            # Ignore modifier keys when pressed alone
            if key_name in ['ctrl', 'alt', 'shift', 'windows']:
                return False
            
            if is_press:
                # Debounce check for press events
                current_time = time.time()
                if current_time - self.last_press_time < self.debounce_interval:
                    return False
                self.last_press_time = current_time
                
                # Get the modifiers that are currently pressed
                modifiers = []
                if keyboard.is_pressed('ctrl'):
                    modifiers.append('ctrl')
                if keyboard.is_pressed('alt'):
                    modifiers.append('alt')
                if keyboard.is_pressed('shift'):
                    modifiers.append('shift')
                if keyboard.is_pressed('windows'):
                    modifiers.append('windows')
                
                # Sort modifiers to ensure consistent order
                modifiers.sort()
                
                # Construct the current hotkey string
                if modifiers:
                    current_hotkey = '+'.join(modifiers + [key_name])
                else:
                    current_hotkey = key_name
                
                # Check if the current combination matches any of our registered hotkeys
                for hotkey_str, config in self.hotkeys.items():
                    if current_hotkey.lower() == hotkey_str.lower():
                        if not config['is_pressed']:  # Only emit if not already pressed
                            print(f"Hotkey press detected: {hotkey_str} (ID: {config['id']})")
                            config['is_pressed'] = True
                            self._on_key_press(event, config['id'])
                            return True
            else:  # Release event
                # Check each registered hotkey
                for hotkey_str, config in self.hotkeys.items():
                    hotkey_parts = hotkey_str.lower().split('+')
                    
                    # For single key hotkeys (like F8)
                    if len(hotkey_parts) == 1 and key_name == hotkey_parts[0]:
                        if config['is_pressed']:
                            print(f"Hotkey release detected: {hotkey_str} (ID: {config['id']})")
                            config['is_pressed'] = False
                            self._on_key_release(event, config['id'])
                            return True
                    # For combination hotkeys (like Ctrl+Space)
                    elif key_name in hotkey_parts:
                        if config['is_pressed']:
                            print(f"Hotkey part release detected: {key_name} in {hotkey_str} (ID: {config['id']})")
                            config['is_pressed'] = False
                            self._on_key_release(event, config['id'])
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error checking hotkey event: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def _check_key_state(self):
        """Fallback mechanism to check if keys are still pressed"""
        try:
            if not self.running:
                return
            
            # Check each hotkey that's in push-to-talk mode
            for hotkey_str, config in self.hotkeys.items():
                if config['is_pressed'] and config['is_push_to_talk'] and platform.system() == "Windows":
                    hotkey_parts = hotkey_str.lower().split('+')
                    
                    # For single key hotkeys
                    if len(hotkey_parts) == 1:
                        if not keyboard.is_pressed(hotkey_parts[0]):
                            print(f"Fallback detection: {hotkey_parts[0]} is no longer pressed (ID: {config['id']})")
                            config['is_pressed'] = False
                            self._on_key_release(None, config['id'])
                    # For combination hotkeys, check if any part is released
                    else:
                        all_pressed = True
                        for part in hotkey_parts:
                            if not keyboard.is_pressed(part):
                                all_pressed = False
                                break
                        if not all_pressed:
                            print(f"Fallback detection: {hotkey_str} combination is no longer pressed (ID: {config['id']})")
                            config['is_pressed'] = False
                            self._on_key_release(None, config['id'])
            
            # Schedule the next check
            if self.running:
                self.key_check_timer = threading.Timer(0.05, self._check_key_state)  # Check every 50ms
                self.key_check_timer.daemon = True
                self.key_check_timer.start()
                
        except Exception as e:
            print(f"Error in key state check: {str(e)}")
            # Schedule the next check even if there was an error
            if self.running:
                self.key_check_timer = threading.Timer(0.05, self._check_key_state)
                self.key_check_timer.daemon = True
                self.key_check_timer.start()

    def backup_clipboard(self):
        """Backup current clipboard content"""
        self.clipboard_manager.backup_clipboard()

    def restore_clipboard(self):
        """Restore clipboard from backup"""
        self.clipboard_manager.restore_clipboard()

    def set_transcription(self, text):
        """Set text for pasting"""
        self.transcription = text

    def paste_transcription(self):
        """Paste the transcription text"""
        if self.transcription:
            try:
                # Set the clipboard to the transcription
                clipboard_set = self.clipboard_manager.set_clipboard_text(self.transcription)
                
                # Log success
                print(f"Transcription copied to clipboard: {self.transcription[:30]}...")
                
                # Try to paste
                paste_success = False
                if clipboard_set:
                    paste_success = self.clipboard_manager.paste_clipboard()
                
                # Emit signal to indicate paste completion status
                self.paste_complete_signal.emit(paste_success)
                print(f"Paste operation completed with status: {paste_success}")
                
                # Return True to indicate that we at least set the clipboard
                return True
                
            except Exception as e:
                print(f"Error during paste operation: {str(e)}")
                self.paste_complete_signal.emit(False)
                # Continue with the process even if paste fails
            
            # Return True to indicate that we at least set the clipboard
            return True
        
        self.paste_complete_signal.emit(False)
        return False

    def run(self):
        """Run the hotkey listener"""
        self.running = True
        
        # Start the fallback timer for key release detection
        self.key_check_timer = threading.Timer(0.1, self._check_key_state)
        self.key_check_timer.daemon = True
        self.key_check_timer.start()
        
        if platform.system() == "Windows":
            self._run_windows_listener()
        else:
            self._run_pynput_listener()
    
    def _run_windows_listener(self):
        """Run the Windows-specific keyboard listener"""
        try:
            print(f"Starting hotkey manager with {len(self.hotkeys)} hotkeys: {list(self.hotkeys.keys())}")
            
            # Create unique callback functions for this manager instance
            def on_press(e):
                # Process events for all hotkeys
                if self._check_hotkey_event(e, True):
                    return True  # Stop event propagation
                return None  # Allow other handlers to process the event
            
            def on_release(e):
                # Process events for all hotkeys
                if self._check_hotkey_event(e, False):
                    return True  # Stop event propagation
                return None  # Allow other handlers to process the event
            
            # Create unique names for the callbacks to prevent overwriting
            callback_id = f"hotkey_manager_{id(self)}"
            on_press.__name__ = f"on_press_{callback_id}"
            on_release.__name__ = f"on_release_{callback_id}"
            
            # Register both press and release handlers
            keyboard.on_press(on_press)
            keyboard.on_release(on_release)
            
            # Keep the thread running
            while self.running:
                time.sleep(0.1)
            
            # Unregister callbacks
            try:
                keyboard.unhook(on_press)
                keyboard.unhook(on_release)
                print("Successfully unhooked keyboard callbacks")
            except Exception as e:
                print(f"Error unhooking keyboard callbacks: {str(e)}")
            
        except Exception as e:
            self.error_signal.emit(f"Hotkey error: {str(e)}")
            print(f"Error in hotkey listener: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _run_pynput_listener(self):
        """Run the pynput keyboard listener for macOS and Linux"""
        try:
            # Convert all hotkeys to pynput key codes
            pynput_keys = {self._convert_hotkey_to_pynput(k): k for k in self.hotkeys.keys()}
            
            def on_press(key):
                for pynput_key, hotkey_str in pynput_keys.items():
                    if (hasattr(key, 'char') and key.char == pynput_key) or key == pynput_key:
                        config = self.hotkeys[hotkey_str]
                        if not config['is_pressed']:
                            config['is_pressed'] = True
                            self._on_key_press(None, config['id'])
            
            def on_release(key):
                for pynput_key, hotkey_str in pynput_keys.items():
                    config = self.hotkeys[hotkey_str]
                    if config['is_push_to_talk'] and ((hasattr(key, 'char') and key.char == pynput_key) or key == pynput_key):
                        if config['is_pressed']:
                            config['is_pressed'] = False
                            self._on_key_release(None, config['id'])
            
            # Start the listener with both press and release callbacks
            listener = None
            try:
                listener = keyboard.Listener(on_press=on_press, on_release=on_release)
                listener.start()
                
                # Keep the thread running
                while self.running:
                    time.sleep(0.1)
                    if listener and not listener.running:
                        break
                        
                # Clean up the listener
                if listener and listener.running:
                    try:
                        listener.stop()
                        print("Successfully stopped pynput listener")
                    except Exception as e:
                        print(f"Error stopping pynput listener: {str(e)}")
            except Exception as e:
                print(f"Error in pynput listener: {str(e)}")
                if listener and listener.running:
                    try:
                        listener.stop()
                    except:
                        pass
                        
        except Exception as e:
            self.error_signal.emit(f"Hotkey error: {str(e)}")
            print(f"Error setting up pynput listener: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _convert_hotkey_to_pynput(self, hotkey):
        """Convert a hotkey string to a pynput key code"""
        # This is a simplified conversion - would need to be expanded for more keys
        if len(hotkey) == 1:
            return keyboard.KeyCode.from_char(hotkey)
        else:
            # Handle special keys
            special_keys = {
                'space': keyboard.Key.space,
                'enter': keyboard.Key.enter,
                'tab': keyboard.Key.tab,
                'esc': keyboard.Key.esc,
                'f1': keyboard.Key.f1,
                'f2': keyboard.Key.f2,
                'f8': keyboard.Key.f8,
                'f9': keyboard.Key.f9,
                # Add more as needed
            }
            return special_keys.get(hotkey.lower(), keyboard.Key.f8)  # Default to F8
    
    def _on_key_press(self, event, hotkey_id):
        """Handle key press event"""
        for hotkey_str, config in self.hotkeys.items():
            if config['id'] == hotkey_id:
                if config['is_push_to_talk']:
                    self.hotkey_press_signal.emit(hotkey_id)
                else:
                    self.hotkey_toggle_signal.emit(hotkey_id)
                break
    
    def _on_key_release(self, event, hotkey_id):
        """Handle key release event"""
        for hotkey_str, config in self.hotkeys.items():
            if config['id'] == hotkey_id and config['is_push_to_talk']:
                self.hotkey_release_signal.emit(hotkey_id)
                break
    
    def stop(self):
        """Stop the hotkey listener"""
        print(f"Stopping hotkey listener with {len(self.hotkeys)} hotkeys")
        self.running = False
        
        # Stop the fallback timer
        if self.key_check_timer:
            try:
                self.key_check_timer.cancel()
            except Exception as e:
                print(f"Error canceling key check timer: {str(e)}")
        
        # Clear the hotkeys dictionary to prevent further processing
        self.hotkeys.clear()

    def add_hotkey(self, hotkey_str, hotkey_id, is_push_to_talk=True):
        """Add or update a hotkey"""
        self.hotkeys[hotkey_str] = {
            'id': hotkey_id,
            'is_push_to_talk': is_push_to_talk,
            'is_pressed': False
        }
        print(f"Added/updated hotkey: {hotkey_str} (ID: {hotkey_id}, Push-to-Talk: {is_push_to_talk})")
    
    def remove_hotkey(self, hotkey_id):
        """Remove a hotkey by ID"""
        for hotkey_str in list(self.hotkeys.keys()):
            if self.hotkeys[hotkey_str]['id'] == hotkey_id:
                del self.hotkeys[hotkey_str]
                print(f"Removed hotkey with ID: {hotkey_id}")
                break
    
    def set_mode(self, hotkey_id, is_push_to_talk):
        """Set push-to-talk or toggle mode for a specific hotkey"""
        for hotkey_str, config in self.hotkeys.items():
            if config['id'] == hotkey_id:
                config['is_push_to_talk'] = is_push_to_talk
                print(f"Set mode for hotkey {hotkey_str} (ID: {hotkey_id}): Push-to-Talk = {is_push_to_talk}")
                break
    
    def force_release(self, hotkey_id):
        """Force release a specific hotkey"""
        for hotkey_str, config in self.hotkeys.items():
            if config['id'] == hotkey_id and config['is_pressed']:
                config['is_pressed'] = False
                self._on_key_release(None, hotkey_id)
                print(f"Forced release of hotkey {hotkey_str} (ID: {hotkey_id})")
                return True
        return False


class HotkeyConfigDialog(QDialog):
    """Dialog for configuring hotkeys"""
    
    def __init__(self, parent=None, current_hotkey=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Hotkey")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add description
        description = QLabel(
            "Press the key combination you want to use as the hotkey.\n"
            "This will be used to start and stop recording."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        
        # Add hotkey edit
        hotkey_layout = QHBoxLayout()
        hotkey_layout.addWidget(QLabel("Hotkey:"))
        
        self.hotkey_edit = QKeySequenceEdit(self)
        if current_hotkey:
            self.hotkey_edit.setKeySequence(QKeySequence(current_hotkey))
        hotkey_layout.addWidget(self.hotkey_edit)
        
        layout.addLayout(hotkey_layout)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.set_button = QPushButton("Set", self)
        self.set_button.clicked.connect(self.accept)
        
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.set_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
    
    def get_hotkey(self):
        """Get the configured hotkey"""
        return self.hotkey_edit.keySequence().toString()


class WhisperUI(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        try:
            super().__init__()
            self.setWindowTitle("Diktando")
            self.setMinimumWidth(800)
            self.setMinimumHeight(600)

            # Initialize variables
            self.current_file = None
            self.is_recording = False
            self.is_transcribing = False
            self.is_hotkey_recording = False
            self.is_llm_recording = False
            self.hotkey = "F8"  # Default transcription hotkey
            self.llm_hotkey = "F9"  # Default LLM hotkey
            
            # Initialize components with error handling
            try:
                print("Initializing LLM processor...")
                self.llm_processor = LLMProcessor()
                print("LLM processor initialized")
            except Exception as e:
                print(f"Error initializing LLM processor: {str(e)}")
                if DEBUG:
                    logger.error(f"Error initializing LLM processor: {str(e)}")
                    logger.error(traceback.format_exc())
                self.llm_processor = None
            
            self.hotkey_manager = None
            
            try:
                print("Initializing clipboard manager...")
                self.clipboard_manager = ClipboardManager()
                print("Clipboard manager initialized")
            except Exception as e:
                print(f"Error initializing clipboard manager: {str(e)}")
                if DEBUG:
                    logger.error(f"Error initializing clipboard manager: {str(e)}")
                    logger.error(traceback.format_exc())
                self.clipboard_manager = None
            
            self.transcription_history = []  # Initialize transcription history
            
            # Connect LLM processor signals with error handling
            if self.llm_processor:
                try:
                    print("Connecting LLM processor signals...")
                    self.llm_processor.processing_complete.connect(self.on_llm_processing_complete)
                    self.llm_processor.processing_error.connect(self.show_error)
                    self.llm_processor.processing_started.connect(lambda: self.show_overlay("Processing with LLM..."))
                    self.llm_processor.processing_stopped.connect(self.hide_overlay)
                    print("LLM processor signals connected")
                except Exception as e:
                    print(f"Error connecting LLM processor signals: {str(e)}")
                    if DEBUG:
                        logger.error(f"Error connecting LLM processor signals: {str(e)}")
                        logger.error(traceback.format_exc())
            else:
                print("LLM processor not initialized, skipping signal connection")
            
            # Initialize UI components
            print("Initializing UI components...")
            self.init_ui()
            print("UI initialized")
            
            print("Initializing models...")
            self.init_models()
            print("Models initialized")
            
            print("Initializing audio devices...")
            self.init_audio_devices()
            print("Audio devices initialized")
            
            # Load settings
            print("Loading settings...")
            self.load_settings()
            print("Settings loaded")
            
            # Enable hotkey by default
            print("Setting up hotkeys...")
            self.enable_hotkey_by_default()
            print("Hotkeys set up")
            
            print("WhisperUI initialization complete")
        except Exception as e:
            print(f"Error during WhisperUI initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def __del__(self):
        """Destructor to clean up resources"""
        print("WhisperUI destructor called, cleaning up resources...")
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up resources before exiting"""
        print("Cleaning up resources...")
        
        # Stop any active recording
        if hasattr(self, 'recorder') and self.recorder and self.recorder.isRunning():
            print("Stopping active recording...")
            self.recorder.recording = False
            self.recorder.wait(2000)  # Wait up to 2 seconds for it to finish
        
        # Stop the hotkey manager
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager and self.hotkey_manager.isRunning():
            print("Stopping hotkey manager...")
            self.hotkey_manager.stop()
            self.hotkey_manager.wait(2000)  # Wait up to 2 seconds for it to finish
        
        # Stop any active transcription
        if hasattr(self, 'transcriber') and self.transcriber and self.transcriber.isRunning():
            print("Stopping active transcription...")
            self.transcriber.wait(2000)  # Wait up to 2 seconds for it to finish
        
        print("Cleanup complete, exiting application")
    
    def create_overlay_window(self):
        """Create an overlay window to show recording status"""
        self.overlay_window = QWidget(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.overlay_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.overlay_window.setFixedSize(180, 60)
        
        # Create main container with rounded corners
        self.overlay_container = QWidget(self.overlay_window)
        self.overlay_container.setObjectName("overlayContainer")
        self.overlay_container.setStyleSheet("""
            #overlayContainer {
                background-color: rgba(28, 28, 30, 180); 
                border-radius: 12px; 
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        self.overlay_container.setGeometry(0, 0, 180, 60)
        
        # Create layout for overlay
        overlay_layout = QVBoxLayout(self.overlay_container)
        overlay_layout.setContentsMargins(12, 12, 12, 12)
        overlay_layout.setSpacing(6)
        
        # Create status layout with icon and label
        status_layout = QHBoxLayout()
        
        # Create animation container
        self.animation_container = QWidget()
        self.animation_container.setFixedSize(16, 16)
        self.animation_container.setStyleSheet("""
            background-color: #FF453A;
            border-radius: 8px;
        """)
        status_layout.addWidget(self.animation_container)
        
        # Add spacing
        status_layout.addSpacing(8)
        
        # Create status label
        self.overlay_label = QLabel("Recording...")
        self.overlay_label.setStyleSheet("""
            color: white; 
            font-size: 14px; 
            font-weight: 500; 
            background-color: transparent;
        """)
        status_layout.addWidget(self.overlay_label)
        status_layout.addStretch()
        
        overlay_layout.addLayout(status_layout)
        
        # Position overlay at bottom-right of screen
        desktop = QApplication.primaryScreen().geometry()
        self.overlay_window.move(
            desktop.width() - self.overlay_window.width() - 20,
            desktop.height() - self.overlay_window.height() - 20
        )
        self.overlay_window.hide()
        
        # Add animation timer
        self.overlay_animation_timer = QTimer(self)
        self.overlay_animation_timer.timeout.connect(self.update_overlay_animation)
        self.overlay_animation_timer.setInterval(800)  # Update every 800ms for a subtle pulse
        
        # Initialize animation state
        self.overlay_animation_value = 0
        self.overlay_animation_direction = 1
    
    def update_overlay_animation(self):
        """Update the overlay animation"""
        if self.overlay_animation_direction > 0:
            self.overlay_animation_value += 20
            if self.overlay_animation_value >= 100:
                self.overlay_animation_direction = -1
        else:
            self.overlay_animation_value -= 20
            if self.overlay_animation_value <= 20:
                self.overlay_animation_direction = 1
        
        # Update the opacity of the animation container
        opacity = 0.5 + (self.overlay_animation_value / 200)  # Range from 0.5 to 1.0
        self.animation_container.setStyleSheet(f"""
            background-color: #FF453A;
            border-radius: 8px;
            opacity: {opacity};
        """)
        
        # Also pulse the size slightly
        size = 14 + (self.overlay_animation_value / 50)  # Range from 14 to 16
        self.animation_container.setFixedSize(int(size), int(size))
        
        # Center the widget to prevent layout shifts
        self.animation_container.setContentsMargins(
            (16 - int(size)) // 2,
            (16 - int(size)) // 2,
            (16 - int(size)) // 2,
            (16 - int(size)) // 2
        )
    
    def show_overlay(self, message):
        """Show the overlay with a message"""
        print(f"Showing overlay with message: {message}")
        self.overlay_label.setText(message)
        
        # Set animation color based on message type
        if "Recording" in message:
            self.animation_container.setStyleSheet("""
                background-color: #FF453A;
                border-radius: 8px;
            """)
        elif "Processing" in message or "Transcribing" in message:
            self.animation_container.setStyleSheet("""
                background-color: #FF9F0A;
                border-radius: 8px;
            """)
        elif "Complete" in message or "Copied" in message or "Pasted" in message:
            self.animation_container.setStyleSheet("""
                background-color: #30D158;
                border-radius: 8px;
            """)
        elif "Error" in message:
            self.animation_container.setStyleSheet("""
                background-color: #FF453A;
                border-radius: 8px;
            """)
        
        # Stop any existing fade timers
        if hasattr(self, 'fade_timer') and self.fade_timer.isActive():
            self.fade_timer.stop()
        if hasattr(self, 'fade_out_timer') and self.fade_out_timer.isActive():
            self.fade_out_timer.stop()
        
        # Set fixed opacity and show immediately
        self.overlay_window.setWindowOpacity(0.95)
        self.overlay_window.show()
        
        # Start animation
        self.overlay_animation_timer.start()
    
    def hide_overlay(self):
        """Hide the overlay window immediately"""
        print("Hiding overlay")
        # Stop any existing fade timers
        if hasattr(self, 'fade_timer') and self.fade_timer.isActive():
            self.fade_timer.stop()
        if hasattr(self, 'fade_out_timer') and self.fade_out_timer.isActive():
            self.fade_out_timer.stop()
        
        # Hide immediately
        self.overlay_window.hide()
        
        # Stop animation
        self.overlay_animation_timer.stop()
    
    def enable_hotkey_by_default(self):
        """Enable the global hotkey by default"""
        try:
            print("Enabling hotkeys by default...")
            if DEBUG:
                logger.info("Enabling hotkeys by default...")
            
            # Check if UI components are initialized
            if not hasattr(self, 'enable_hotkey_check'):
                print("Warning: enable_hotkey_check not initialized yet, skipping hotkey setup")
                if DEBUG:
                    logger.warning("enable_hotkey_check not initialized yet, skipping hotkey setup")
                return
            
            try:
                self.enable_hotkey_check.setChecked(True)
            except Exception as e:
                print(f"Error setting enable_hotkey_check: {str(e)}")
                if DEBUG:
                    logger.error(f"Error setting enable_hotkey_check: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Make sure we have default hotkeys set
            if not self.hotkey:
                self.hotkey = 'F8'
                if hasattr(self, 'hotkey_label'):
                    try:
                        self.hotkey_label.setText(self.hotkey)
                    except Exception as e:
                        print(f"Error setting hotkey_label text: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error setting hotkey_label text: {str(e)}")
                else:
                    print("Warning: hotkey_label not initialized yet")
                    if DEBUG:
                        logger.warning("hotkey_label not initialized yet")
            
            if not self.llm_hotkey:
                self.llm_hotkey = 'F9'
                if hasattr(self, 'llm_hotkey_label'):
                    try:
                        self.llm_hotkey_label.setText(self.llm_hotkey)
                    except Exception as e:
                        print(f"Error setting llm_hotkey_label text: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error setting llm_hotkey_label text: {str(e)}")
                else:
                    print("Warning: llm_hotkey_label not initialized yet")
                    if DEBUG:
                        logger.warning("llm_hotkey_label not initialized yet")
            
            # Enable the hotkeys with error handling
            try:
                self.toggle_hotkey(True)
                print("Hotkeys enabled successfully")
                if DEBUG:
                    logger.info("Hotkeys enabled successfully")
            except Exception as e:
                print(f"Error enabling hotkeys: {str(e)}")
                if DEBUG:
                    logger.error(f"Error enabling hotkeys: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Try to provide more diagnostic information
                if hasattr(self, 'hotkey_manager'):
                    print(f"Hotkey manager exists: {self.hotkey_manager is not None}")
                    if DEBUG:
                        logger.info(f"Hotkey manager exists: {self.hotkey_manager is not None}")
                else:
                    print("Hotkey manager not initialized")
                    if DEBUG:
                        logger.warning("Hotkey manager not initialized")
        
        except Exception as e:
            print(f"Error in enable_hotkey_by_default: {str(e)}")
            if DEBUG:
                logger.error(f"Error in enable_hotkey_by_default: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Try to initialize hotkeys in a fallback way
            try:
                print("Attempting fallback hotkey initialization...")
                if DEBUG:
                    logger.info("Attempting fallback hotkey initialization...")
                
                self.hotkey = 'F8'
                self.llm_hotkey = 'F9'
                
                # Only try to initialize hotkey manager if it doesn't exist
                if not hasattr(self, 'hotkey_manager') or self.hotkey_manager is None:
                    self.hotkey_manager = HotkeyManager(self)
                    self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
                    self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
                    self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)
                    self.hotkey_manager.error_signal.connect(self.show_error)
                
                # Add the hotkeys
                self.hotkey_manager.add_hotkey(self.hotkey, "transcription", True)
                self.hotkey_manager.add_hotkey(self.llm_hotkey, "llm", True)
                
                # Start the hotkey manager
                if not self.hotkey_manager.isRunning():
                    self.hotkey_manager.start()
                
                print("Fallback hotkey initialization successful")
                if DEBUG:
                    logger.info("Fallback hotkey initialization successful")
            except Exception as fallback_error:
                print(f"Fallback hotkey initialization failed: {str(fallback_error)}")
                if DEBUG:
                    logger.error(f"Fallback hotkey initialization failed: {str(fallback_error)}")
                    logger.error(traceback.format_exc())
                print("Hotkey initialization failed, continuing without hotkeys")
    
    def init_ui(self):
        """Initialize the user interface"""
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
            # Set the application-wide icon
            QApplication.setWindowIcon(icon)
            # Also set the taskbar icon for Windows
            if platform.system() == "Windows":
                import ctypes
                myappid = u'diktando.speech.recognition.1.1'  # arbitrary string
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing...")

        # Create overlay window
        self.create_overlay_window()

        # Initialize managers and handlers
        self.clipboard_manager = ClipboardManager()
        self.hotkey_manager = HotkeyManager(self)
        self.hotkey_manager.hotkey_toggle_signal.connect(self.on_hotkey_toggle)
        self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
        self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
        self.hotkey_manager.error_signal.connect(self.show_error)
        self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)

        # Initialize update checker
        self.update_checker = UpdateChecker()
        self.update_checker.update_available_signal.connect(self.show_update_dialog)
        self.update_checker.update_progress_signal.connect(self.update_progress)
        self.update_checker.update_error_signal.connect(self.show_error)
        self.update_checker.update_success_signal.connect(self.handle_update_success)

        # Initialize timers
        self.hotkey_recording_timer = QTimer(self)
        self.hotkey_recording_timer.timeout.connect(self.update_hotkey_recording_indicator)
        self.hotkey_recording_timer.setInterval(500)  # Update every 500ms

        # Create main tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Create tabs
        transcription_tab = QWidget()
        model_tab = QWidget()
        settings_tab = QWidget()
        
        # Setup tab layouts
        self.setup_transcription_tab(transcription_tab)
        self.setup_model_tab(model_tab)
        self.setup_settings_tab(settings_tab)
        
        # Add tabs to widget
        self.tab_widget.addTab(transcription_tab, "Transcription")
        self.tab_widget.addTab(model_tab, "Models")
        self.tab_widget.addTab(settings_tab, "Settings")

        # Initialize recording paths
        self.recording_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.wav")
        self.debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_recordings")
        self.silence_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silence.wav")
        os.makedirs(self.debug_dir, exist_ok=True)

        # Generate silence file if it doesn't exist
        if not os.path.exists(self.silence_path):
            self.generate_silence_file()

        # Show the main window
        self.show()

        # Check for updates if enabled
        if hasattr(self, 'auto_check_updates') and self.auto_check_updates.isChecked():
            QTimer.singleShot(1000, lambda: self.update_checker.check_for_updates(silent=True))
    
    def setup_transcription_tab(self, tab):
        """Set up the transcription tab"""
        
        layout = QVBoxLayout(tab)
        
        # Audio source selection
        source_group = QGroupBox("Audio Source")
        source_layout = QHBoxLayout(source_group)
        
        self.mic_radio = QRadioButton("Microphone")
        self.mic_radio.setChecked(True)
        self.mic_radio.toggled.connect(self.toggle_audio_source)
        
        self.file_radio = QRadioButton("File")
        self.file_radio.toggled.connect(self.toggle_audio_source)
        
        source_layout.addWidget(self.mic_radio)
        source_layout.addWidget(self.file_radio)
        
        layout.addWidget(source_group)
        
        # Microphone controls
        self.mic_group = QGroupBox("Microphone Recording")
        mic_layout = QVBoxLayout(self.mic_group)
        
        mic_controls_layout = QHBoxLayout()
        
        self.device_combo = QComboBox()
        mic_controls_layout.addWidget(QLabel("Audio Device:"))
        mic_controls_layout.addWidget(self.device_combo)
        
        # Add sample rate combo box
        mic_controls_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000", "44100", "48000"])
        mic_controls_layout.addWidget(self.sample_rate_combo)
        
        # Add channels combo box
        mic_controls_layout.addWidget(QLabel("Channels:"))
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["Mono", "Stereo"])
        mic_controls_layout.addWidget(self.channels_combo)
        
        mic_layout.addLayout(mic_controls_layout)
        
        record_layout = QHBoxLayout()
        
        self.record_button = QPushButton("Record")
        self.record_button.clicked.connect(self.toggle_recording)
        record_layout.addWidget(self.record_button)
        
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setValue(0)
        record_layout.addWidget(self.level_meter)
        
        mic_layout.addLayout(record_layout)
        
        layout.addWidget(self.mic_group)
        
        # File selection controls
        self.file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout(self.file_group)
        
        self.file_path = QLabel("No file selected")
        file_layout.addWidget(self.file_path)
        
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_button)
        
        layout.addWidget(self.file_group)
        self.file_group.setVisible(False)
        
        # Transcription controls
        trans_group = QGroupBox("Transcription")
        trans_layout = QVBoxLayout(trans_group)
        
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        controls_layout.addWidget(self.model_combo)
        
        controls_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["en", "auto", "de", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"])
        controls_layout.addWidget(self.language_combo)
        
        trans_layout.addLayout(controls_layout)
        
        self.transcribe_button = QPushButton("Transcribe")
        self.transcribe_button.clicked.connect(self.start_transcription)
        trans_layout.addWidget(self.transcribe_button)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        trans_layout.addWidget(self.progress_bar)
        
        layout.addWidget(trans_group)
        
        # Results area
        results_group = QGroupBox("Transcription Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        buttons_layout.addWidget(copy_button)
        
        # Add export button with menu
        export_button = QPushButton("Export")
        export_menu = QMenu(export_button)
        export_menu.addAction("Export as Text", self.export_as_text)
        export_menu.addAction("Export as PDF", self.export_as_pdf)
        export_menu.addAction("Export as SRT", self.export_as_srt)
        export_button.setMenu(export_menu)
        buttons_layout.addWidget(export_button)
        
        clear_button = QPushButton("Clear History")
        clear_button.clicked.connect(self.clear_transcription_history)
        buttons_layout.addWidget(clear_button)
        
        results_layout.addLayout(buttons_layout)
        
        layout.addWidget(results_group)
    
    def setup_model_tab(self, tab):
        """Set up the model management tab"""
        
        layout = QVBoxLayout(tab)
        
        # Available models
        available_group = QGroupBox("Available Models")
        available_layout = QVBoxLayout(available_group)
        
        self.models_combo = QComboBox()
        self.models_combo.addItems([
            "tiny", "tiny.en", "base", "base.en", 
            "small", "small.en", "medium", "medium.en", "large-v1"
        ])
        
        # Set base.en as the default model
        index = self.models_combo.findText("base.en")
        if index >= 0:
            self.models_combo.setCurrentIndex(index)
            
        available_layout.addWidget(self.models_combo)
        
        download_button = QPushButton("Download Selected Model")
        download_button.clicked.connect(self.download_model)
        available_layout.addWidget(download_button)
        
        self.download_progress = QProgressBar()
        self.download_progress.setRange(0, 100)
        self.download_progress.setValue(0)
        available_layout.addWidget(self.download_progress)
        
        layout.addWidget(available_group)
        
        # Installed models
        installed_group = QGroupBox("Installed Models")
        installed_layout = QVBoxLayout(installed_group)
        
        self.installed_list = QTextEdit()
        self.installed_list.setReadOnly(True)
        installed_layout.addWidget(self.installed_list)
        
        refresh_button = QPushButton("Refresh List")
        refresh_button.clicked.connect(self.refresh_models)
        installed_layout.addWidget(refresh_button)
        
        layout.addWidget(installed_group)
    
    def setup_settings_tab(self, tab):
        layout = QVBoxLayout(tab)
        
        # Hotkey settings
        hotkey_group = QGroupBox("Hotkey Settings")
        hotkey_layout = QVBoxLayout(hotkey_group)
        
        # Enable hotkey checkbox
        self.enable_hotkey_check = QCheckBox("Enable Global Hotkey")
        self.enable_hotkey_check.toggled.connect(self.toggle_hotkey)
        hotkey_layout.addWidget(self.enable_hotkey_check)
        
        # Mode selection
        mode_group = QButtonGroup(self)
        self.push_to_talk_radio = QRadioButton("Push-to-Talk Mode")
        self.toggle_mode_radio = QRadioButton("Toggle Mode")
        mode_group.addButton(self.push_to_talk_radio)
        mode_group.addButton(self.toggle_mode_radio)
        
        # Set push-to-talk as default
        self.push_to_talk_radio.setChecked(True)
        self.toggle_mode_radio.setChecked(False)
        
        mode_group.buttonClicked.connect(self.on_mode_changed)
        hotkey_layout.addWidget(self.push_to_talk_radio)
        hotkey_layout.addWidget(self.toggle_mode_radio)
        
        # Hotkey configuration
        hotkey_config_layout = QHBoxLayout()
        hotkey_config_layout.addWidget(QLabel("Hotkey:"))
        self.hotkey_label = QLabel("f8")
        hotkey_config_layout.addWidget(self.hotkey_label)
        
        configure_button = QPushButton("Configure")
        configure_button.clicked.connect(self.show_hotkey_config)
        hotkey_config_layout.addWidget(configure_button)
        hotkey_config_layout.addStretch()
        
        hotkey_layout.addLayout(hotkey_config_layout)
        
        # Add auto-paste option
        self.auto_paste_check = QCheckBox("Automatically paste transcription (may require permissions)")
        self.auto_paste_check.setChecked(True)
        hotkey_layout.addWidget(self.auto_paste_check)
        
        layout.addWidget(hotkey_group)
        
        # Application settings
        app_group = QGroupBox("Application Settings")
        app_layout = QVBoxLayout(app_group)
        
        self.dark_mode_check = QCheckBox("Dark Mode")
        self.dark_mode_check.toggled.connect(self.toggle_dark_mode)
        app_layout.addWidget(self.dark_mode_check)
        
        models_dir_layout = QHBoxLayout()
        models_dir_layout.addWidget(QLabel("Models Directory:"))
        self.models_dir_path = QLabel(self.get_models_dir())
        models_dir_layout.addWidget(self.models_dir_path)
        
        change_dir_button = QPushButton("Change")
        change_dir_button.clicked.connect(self.change_models_dir)
        models_dir_layout.addWidget(change_dir_button)
        
        app_layout.addLayout(models_dir_layout)
        
        layout.addWidget(app_group)
        
        # Update settings
        update_group = QGroupBox("Updates")
        update_layout = QVBoxLayout(update_group)
        
        # Add check for updates button
        check_updates_btn = QPushButton("Check for Updates")
        check_updates_btn.clicked.connect(lambda: self.update_checker.check_for_updates(silent=False))
        update_layout.addWidget(check_updates_btn)
        
        # Add auto-check for updates option
        self.auto_check_updates = QCheckBox("Check for updates on startup")
        self.auto_check_updates.setChecked(True)
        update_layout.addWidget(self.auto_check_updates)
        
        layout.addWidget(update_group)
        
        # About section
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        
        about_text = QLabel(
        "Diktando\n"
            "A Windows GUI for OpenAI's Whisper speech recognition model\n"
            "Using whisper-cpp for local transcription\n"
            f"Version: {self.update_checker.CURRENT_VERSION}\n"
        )
        about_text.setAlignment(Qt.AlignCenter)
        about_layout.addWidget(about_text)
        
        github_button = QPushButton("Visit GitHub Repository")
        github_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/fspecii/diktando")
        ))
        about_layout.addWidget(github_button)
        
        layout.addWidget(about_group)
        
        # Add LLM settings section
        llm_group = QGroupBox("LLM Processing")
        llm_layout = QVBoxLayout()
        
        # LLM hotkey configuration
        llm_hotkey_layout = QHBoxLayout()
        llm_hotkey_layout.addWidget(QLabel("LLM Hotkey:"))
        self.llm_hotkey_label = QLabel(self.llm_hotkey if hasattr(self, 'llm_hotkey') and self.llm_hotkey else "F9")
        llm_hotkey_layout.addWidget(self.llm_hotkey_label)
        
        llm_configure_button = QPushButton("Configure")
        llm_configure_button.clicked.connect(lambda: self.show_llm_settings(show_hotkey_tab=True))
        llm_hotkey_layout.addWidget(llm_configure_button)
        llm_hotkey_layout.addStretch()
        
        llm_layout.addLayout(llm_hotkey_layout)
        
        # LLM settings button
        llm_settings_btn = QPushButton("Configure LLM Settings")
        llm_settings_btn.clicked.connect(self.show_llm_settings)
        llm_layout.addWidget(llm_settings_btn)
        
        llm_group.setLayout(llm_layout)
        layout.addWidget(llm_group)
    
    def init_models(self):
        """Initialize the models list"""
        self.refresh_models()
    
    def init_audio_devices(self):
        """Initialize the audio devices list"""
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        for i, device in enumerate(input_devices):
            self.device_combo.addItem(f"{device['name']}", i)
    
    def toggle_audio_source(self):
        """Toggle between microphone and file input"""
        self.mic_group.setVisible(self.mic_radio.isChecked())
        self.file_group.setVisible(self.file_radio.isChecked())
    
    def toggle_recording(self):
        """Start or stop recording"""
        if self.record_button.text() == "Record":
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio"""
        try:
            device_idx = self.device_combo.currentData()
            self.sample_rate = int(self.sample_rate_combo.currentText())  # Store sample_rate as instance variable
            channels = 1 if self.channels_combo.currentIndex() == 0 else 2
            
            self.recorder = AudioRecorder(self.sample_rate, channels, device_idx)
            self.recorder.update_signal.connect(self.update_level_meter)
            self.recorder.finished_signal.connect(self.recording_finished)
            self.recorder.error_signal.connect(self.show_error)
            
            self.recorder.start_recording()
            self.is_recording = True  # Set recording flag
            
            self.record_button.setText("Stop")
            self.status_bar.showMessage("Recording...")
            
        except Exception as e:
            self.show_error(f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        """Stop recording audio"""
        if self.recorder and self.recorder.recording:
            print("Stopping active recording...")
            self.recorder.stop_recording()
            self.is_recording = False  # Reset recording flag
            self.record_button.setText("Record")
            self.status_bar.showMessage("Processing recording...")
    
    def update_level_meter(self, data):
        """Update the audio level meter"""
        try:
            # Calculate RMS value and convert to percentage
            rms = np.sqrt(np.mean(np.square(data)))
            level = min(int(rms * 100), 100)  # Clamp to 0-100 range
            self.level_meter.setValue(level)
        except Exception as e:
            print(f"Error updating level meter: {str(e)}")
            # Set to 0 in case of error
            self.level_meter.setValue(0)
    
    def recording_finished(self):
        """Called when recording is finished"""
        try:
            if not self.recorder or not self.recorder.audio_data:
                self.log_message("No audio data available")
                return

            # Convert audio data to numpy array and ensure it's 1D
            audio_array = np.concatenate([chunk.flatten() for chunk in self.recorder.audio_data])
            
            # Calculate duration in milliseconds
            duration_ms = len(audio_array) / self.sample_rate * 1000
            self.log_message(f"Recording duration: {duration_ms:.0f}ms")
            
            # Create debug directory if it doesn't exist
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_recordings")
            os.makedirs(debug_dir, exist_ok=True)
            
            # Save a copy if recording is too short
            if duration_ms < 1000:
                debug_path = os.path.join(debug_dir, f"short_recording_{int(time.time())}.wav")
                sf.write(debug_path, audio_array, self.sample_rate)
                self.log_message(f"Saved short recording for debugging: {debug_path}")
                
                # Pad audio if shorter than 1000ms
                samples_needed = int((1000 - duration_ms) * self.sample_rate / 1000)
                padding = np.zeros(samples_needed, dtype=audio_array.dtype)
                audio_array = np.concatenate([audio_array, padding])
                self.log_message(f"Added {samples_needed} samples of silence padding")

            # Check audio levels
            min_val = np.min(audio_array)
            max_val = np.max(audio_array)
            self.log_message(f"Audio array shape: {audio_array.shape}, min: {min_val}, max: {max_val}")

            # Save a copy if audio is very quiet
            max_amplitude = max(abs(min_val), abs(max_val))
            if max_amplitude < 0.1:
                debug_path = os.path.join(debug_dir, f"low_volume_{int(time.time())}.wav")
                sf.write(debug_path, audio_array, self.sample_rate)
                self.log_message(f"Saved low volume recording for debugging: {debug_path}")
                
                self.log_message(f"Audio signal is very quiet (max amplitude: {max_amplitude}), normalizing")
                audio_array = audio_array / max_amplitude * 0.9

            # Save to WAV file
            self.log_message(f"Saving audio to {self.recording_path}")
            sf.write(self.recording_path, audio_array, self.sample_rate)
            file_size = os.path.getsize(self.recording_path)
            self.log_message(f"Audio file saved successfully: {file_size} bytes")

            # Start transcription
            self.hotkey_recording_finished(self.recording_path)

        except Exception as e:
            self.log_message(f"Error in recording_finished: {str(e)}")
            self.show_error(f"Error processing recording: {str(e)}")
    
    def browse_file(self):
        """Browse for an audio file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.wav *.mp3 *.ogg *.flac *.m4a);;All Files (*)"
        )
        
        if file_path:
            self.file_path.setText(file_path)
    
    def start_transcription(self):
        """Start the transcription process"""
        try:
            # Get the audio file
            if self.mic_radio.isChecked():
                audio_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.wav")
                if not os.path.exists(audio_file):
                    self.show_error("No recording found. Please record audio first.")
                    return
            else:
                audio_file = self.file_path.text()
                if audio_file == "No file selected" or not os.path.exists(audio_file):
                    self.show_error("Please select a valid audio file.")
                    return
            
            # Get the model
            model_name = self.model_combo.currentText()
            model_path = os.path.join(self.get_models_dir(), f"ggml-{model_name}.bin")
            if not os.path.exists(model_path):
                self.show_error(f"Model not found: {model_path}. Please download it first.")
                return
            
            # Get the language
            language = self.language_combo.currentText()
            
            # Start transcription
            self.transcribe_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_bar.showMessage("Transcribing...")
            
            self.transcriber = Transcriber(audio_file, model_path, language)
            self.transcriber.progress_signal.connect(self.update_transcription_progress)
            self.transcriber.finished_signal.connect(self.transcription_finished)
            self.transcriber.error_signal.connect(self.show_error)
            
            self.transcriber.start()
            
        except Exception as e:
            self.show_error(f"Failed to start transcription: {str(e)}")
    
    def update_transcription_progress(self, progress):
        """Update the transcription progress bar"""
        self.progress_bar.setValue(progress)
    
    def transcription_finished(self, text):
        """Handle the finished transcription"""
        # Add to history and get cleaned text
        cleaned_text = self.append_to_transcription_history(text, source="file")
        
        self.transcribe_button.setEnabled(True)
        self.progress_bar.setValue(100)
        self.status_bar.showMessage("Transcription complete")
    
    def copy_to_clipboard(self):
        """Copy the transcription results to the clipboard"""
        text = self.results_text.toPlainText()
        
        # Clean the transcription before copying
        cleaned_text = self.clean_transcription(text)
        
        # Use the clipboard manager to set the text
        self.clipboard_manager.set_clipboard_text(cleaned_text)
        self.status_bar.showMessage("Copied to clipboard")
    
    def download_model(self):
        """Download the selected model"""
        try:
            model_name = self.models_combo.currentText()
            model_url = self.get_model_url(model_name)
            output_dir = self.get_models_dir()
            
            self.status_bar.showMessage(f"Downloading model: {model_name}")
            self.download_progress.setValue(0)
            
            self.downloader = ModelDownloader(model_name, model_url, output_dir)
            self.downloader.progress_signal.connect(self.update_download_progress)
            self.downloader.finished_signal.connect(self.download_finished)
            self.downloader.error_signal.connect(self.handle_download_error)
            
            # Set the download flag
            self.is_download_in_progress = True
            
            self.downloader.start()
            
        except Exception as e:
            self.is_download_in_progress = False
            self.show_error(f"Failed to start download: {str(e)}")
    
    def update_download_progress(self, progress):
        """Update the download progress bar"""
        self.download_progress.setValue(progress)
    
    def download_finished(self):
        """Handle the completion of a model download"""
        # Reset the download flag
        self.is_download_in_progress = False
        
        self.status_bar.showMessage("Download complete")
        self.download_progress.setValue(100)
        self.refresh_models()
        
        # Show a message to the user
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Download Complete")
        msg.setText("Model download completed successfully.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
    
    def refresh_models(self):
        """Refresh the list of installed models"""
        models_dir = self.get_models_dir()
        
        if not os.path.exists(models_dir):
            os.makedirs(models_dir, exist_ok=True)
        
        model_files = []
        for file in os.listdir(models_dir):
            if file.startswith("ggml-") and file.endswith(".bin"):
                model_name = file[5:-4]  # Remove "ggml-" prefix and ".bin" suffix
                file_path = os.path.join(models_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                model_files.append(f"{model_name} ({size_mb:.1f} MB)")
        
        self.installed_list.setPlainText("\n".join(model_files) if model_files else "No models installed")
        
        # Update the model combo box in the transcription tab
        current_model = self.model_combo.currentText()
        self.model_combo.clear()
        
        available_models = []
        for file in os.listdir(models_dir):
            if file.startswith("ggml-") and file.endswith(".bin"):
                model_name = file[5:-4]  # Remove "ggml-" prefix and ".bin" suffix
                available_models.append(model_name)
                self.model_combo.addItem(model_name)
        
        # Try to restore the previously selected model
        if current_model and current_model in available_models:
            index = self.model_combo.findText(current_model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        # Otherwise, prefer base.en if available
        elif "base.en" in available_models:
            index = self.model_combo.findText("base.en")
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        # If no base.en, try any English model
        elif any(model.endswith(".en") for model in available_models):
            for model in available_models:
                if model.endswith(".en"):
                    index = self.model_combo.findText(model)
                    if index >= 0:
                        self.model_combo.setCurrentIndex(index)
                    break
    
    def get_models_dir(self):
        """Get the models directory"""
        home_dir = str(Path.home())
        models_dir = os.path.join(home_dir, "whisper-ui")
        
        # Create the directory if it doesn't exist
        if not os.path.exists(models_dir):
            try:
                os.makedirs(models_dir, exist_ok=True)
                print(f"Created models directory: {models_dir}")
            except Exception as e:
                print(f"Failed to create models directory: {str(e)}")
        
        print(f"Using models directory: {models_dir}")
        return models_dir
    
    def change_models_dir(self):
        """Change the models directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Models Directory", self.get_models_dir()
        )
        
        if dir_path:
            # TODO: Implement settings storage
            self.models_dir_path.setText(dir_path)
            self.refresh_models()
    
    def toggle_dark_mode(self, enabled):
        """Toggle dark mode"""
        if enabled:
            # Dark mode stylesheet
            self.setStyleSheet("""
                QWidget {
                    background-color: #2D2D30;
                    color: #E0E0E0;
                }
                QMainWindow, QDialog {
                    background-color: #1E1E1E;
                }
                QPushButton {
                    background-color: #0E639C;
                    color: white;
                    border: 1px solid #0E639C;
                    border-radius: 4px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #1177BB;
                }
                QPushButton:pressed {
                    background-color: #0D5A8E;
                }
                QPushButton:disabled {
                    background-color: #3D3D3D;
                    color: #9D9D9D;
                    border: 1px solid #3D3D3D;
                }
                QLineEdit, QTextEdit, QComboBox, QSpinBox {
                    background-color: #3D3D3D;
                    color: #E0E0E0;
                    border: 1px solid #5D5D5D;
                    border-radius: 4px;
                    padding: 2px;
                }
                QComboBox::drop-down {
                    border: 0px;
                }
                QComboBox::down-arrow {
                    image: url(down_arrow.png);
                    width: 12px;
                    height: 12px;
                }
                QProgressBar {
                    border: 1px solid #5D5D5D;
                    border-radius: 4px;
                    background-color: #3D3D3D;
                    text-align: center;
                    color: white;
                }
                QProgressBar::chunk {
                    background-color: #0E639C;
                    width: 10px;
                    margin: 0.5px;
                }
                QGroupBox {
                    border: 1px solid #5D5D5D;
                    border-radius: 4px;
                    margin-top: 10px;
                    padding-top: 10px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 5px;
                }
                QTabWidget::pane {
                    border: 1px solid #5D5D5D;
                    border-radius: 4px;
                }
                QTabBar::tab {
                    background-color: #2D2D30;
                    color: #E0E0E0;
                    border: 1px solid #5D5D5D;
                    border-bottom-color: #5D5D5D;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                    min-width: 8ex;
                    padding: 5px;
                }
                QTabBar::tab:selected {
                    background-color: #0E639C;
                    border-bottom-color: #0E639C;
                }
                QTabBar::tab:!selected {
                    margin-top: 2px;
                }
                QCheckBox, QRadioButton {
                    color: #E0E0E0;
                }
                QCheckBox::indicator, QRadioButton::indicator {
                    width: 13px;
                    height: 13px;
                }
                QStatusBar {
                    background-color: #007ACC;
                    color: white;
                }
                QMenuBar {
                    background-color: #2D2D30;
                    color: #E0E0E0;
                }
                QMenuBar::item {
                    background: transparent;
                }
                QMenuBar::item:selected {
                    background-color: #3D3D3D;
                }
                QMenu {
                    background-color: #2D2D30;
                    color: #E0E0E0;
                    border: 1px solid #5D5D5D;
                }
                QMenu::item:selected {
                    background-color: #3D3D3D;
                }
            """)
        else:
            # Light mode (default Qt style)
            self.setStyleSheet("")
        
        # Save the setting
        self.save_settings()
        
        # Update status
        mode_str = "Dark" if enabled else "Light"
        self.status_bar.showMessage(f"Switched to {mode_str} mode")
    
    def get_model_url(self, model_name):
        """Get the URL for the specified model"""
        base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
        return f"{base_url}/ggml-{model_name}.bin"
    
    def show_error(self, message):
        """Show an error message in the UI"""
        print(f"Error: {message}")
        
        # Show error in overlay
        if hasattr(self, 'overlay_window'):
            self.show_overlay(f"Error: {message}")
        
        # Show error in status bar if available
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(f"Error: {message}")
        
        # Show error dialog
        QMessageBox.critical(self, "Error", message)
        
        # Also show in overlay
        self.show_overlay("ERROR: " + message)
        QTimer.singleShot(3000, self.hide_overlay)

    def toggle_hotkey(self, enabled):
        """Toggle the global hotkey functionality"""
        try:
            if enabled:
                # Get the current hotkey
                try:
                    hotkey = self.hotkey_label.text()
                    if not hotkey:
                        print("No hotkey configured")
                        if DEBUG:
                            logger.warning("No hotkey configured")
                        self.show_error("Please configure a hotkey first")
                        self.enable_hotkey_check.setChecked(False)
                        return
                except Exception as e:
                    print(f"Error getting hotkey text: {str(e)}")
                    if DEBUG:
                        logger.error(f"Error getting hotkey text: {str(e)}")
                        logger.error(traceback.format_exc())
                    self.enable_hotkey_check.setChecked(False)
                    return
                
                try:
                    # Convert QKeySequence to a string format that keyboard library can understand
                    hotkey_str = self._convert_key_sequence_to_string(QKeySequence(hotkey))
                    print(f"Using hotkey: {hotkey_str}")
                    if DEBUG:
                        logger.info(f"Using hotkey: {hotkey_str}")
                    
                    # Initialize the hotkey manager if it doesn't exist
                    if not hasattr(self, 'hotkey_manager') or not self.hotkey_manager:
                        print("Creating new hotkey manager")
                        if DEBUG:
                            logger.info("Creating new hotkey manager")
                        self.hotkey_manager = HotkeyManager(self)
                        
                        # Connect signals
                        try:
                            print("Connecting hotkey manager signals")
                            if DEBUG:
                                logger.info("Connecting hotkey manager signals")
                            self.hotkey_manager.hotkey_toggle_signal.connect(self.on_hotkey_toggle)
                            self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
                            self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
                            self.hotkey_manager.error_signal.connect(self.show_error)
                            self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)
                        except Exception as e:
                            print(f"Error connecting hotkey manager signals: {str(e)}")
                            if DEBUG:
                                logger.error(f"Error connecting hotkey manager signals: {str(e)}")
                                logger.error(traceback.format_exc())
                            raise
                    
                    # Stop the manager if it's running
                    if self.hotkey_manager.isRunning():
                        print("Stopping existing hotkey manager")
                        if DEBUG:
                            logger.info("Stopping existing hotkey manager")
                        try:
                            self.hotkey_manager.stop()
                            self.hotkey_manager.wait(2000)  # Wait up to 2 seconds for thread to finish
                        except Exception as e:
                            print(f"Error stopping existing hotkey manager: {str(e)}")
                            if DEBUG:
                                logger.error(f"Error stopping existing hotkey manager: {str(e)}")
                                logger.error(traceback.format_exc())
                    
                    # Add or update the transcription hotkey
                    try:
                        is_push_to_talk = self.push_to_talk_radio.isChecked()
                        print(f"Adding/updating hotkey: {hotkey_str} (ID: transcription, Push-to-Talk: {is_push_to_talk})")
                        if DEBUG:
                            logger.info(f"Adding/updating hotkey: {hotkey_str} (ID: transcription, Push-to-Talk: {is_push_to_talk})")
                        self.hotkey_manager.add_hotkey(hotkey_str, "transcription", is_push_to_talk)
                    except Exception as e:
                        print(f"Error adding transcription hotkey: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error adding transcription hotkey: {str(e)}")
                            logger.error(traceback.format_exc())
                        raise
                    
                    # Add LLM hotkey if configured
                    if hasattr(self, 'llm_hotkey') and self.llm_hotkey:
                        try:
                            llm_hotkey_str = self._convert_key_sequence_to_string(QKeySequence(self.llm_hotkey))
                            print(f"Adding/updating hotkey: {llm_hotkey_str} (ID: llm, Push-to-Talk: {is_push_to_talk})")
                            if DEBUG:
                                logger.info(f"Adding/updating hotkey: {llm_hotkey_str} (ID: llm, Push-to-Talk: {is_push_to_talk})")
                            self.hotkey_manager.add_hotkey(llm_hotkey_str, "llm", is_push_to_talk)
                        except Exception as e:
                            print(f"Error adding LLM hotkey: {str(e)}")
                            if DEBUG:
                                logger.error(f"Error adding LLM hotkey: {str(e)}")
                                logger.error(traceback.format_exc())
                    
                    # Start the manager
                    try:
                        print("Starting hotkey manager with hotkeys")
                        if DEBUG:
                            logger.info("Starting hotkey manager with hotkeys")
                        self.hotkey_manager.start()
                    except Exception as e:
                        print(f"Error starting hotkey manager: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error starting hotkey manager: {str(e)}")
                            logger.error(traceback.format_exc())
                        raise
                    
                    # Update status
                    try:
                        mode_str = "Push-to-Talk" if is_push_to_talk else "Toggle"
                        self.status_bar.showMessage(f"Hotkey enabled: {hotkey} ({mode_str} mode)")
                    except Exception as e:
                        print(f"Error updating status bar: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error updating status bar: {str(e)}")
                
                except Exception as e:
                    print(f"Failed to enable hotkey: {str(e)}")
                    if DEBUG:
                        logger.error(f"Failed to enable hotkey: {str(e)}")
                        logger.error(traceback.format_exc())
                    self.show_error(f"Failed to enable hotkey: {str(e)}")
                    self.enable_hotkey_check.setChecked(False)
                    return
            else:
                # Stop the hotkey manager
                if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                    try:
                        print("Stopping hotkey manager")
                        if DEBUG:
                            logger.info("Stopping hotkey manager")
                        self.hotkey_manager.stop()
                        self.hotkey_manager.wait(2000)  # Wait up to 2 seconds for thread to finish
                        
                        # Clean up any ongoing recording
                        if self.is_hotkey_recording:
                            self.stop_hotkey_recording()
                        
                        # Reset state
                        self.is_hotkey_recording = False
                        
                    except Exception as e:
                        print(f"Error stopping hotkey manager: {str(e)}")
                        if DEBUG:
                            logger.error(f"Error stopping hotkey manager: {str(e)}")
                            logger.error(traceback.format_exc())
                    finally:
                        try:
                            self.status_bar.showMessage("Hotkey disabled")
                        except Exception as e:
                            print(f"Error updating status bar: {str(e)}")
                            if DEBUG:
                                logger.error(f"Error updating status bar: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in toggle_hotkey: {str(e)}")
            if DEBUG:
                logger.error(f"Unexpected error in toggle_hotkey: {str(e)}")
                logger.error(traceback.format_exc())
            try:
                self.show_error(f"Unexpected error in toggle_hotkey: {str(e)}")
            except:
                pass

    def setup_llm_hotkey(self):
        """Setup LLM hotkey in the unified hotkey manager"""
        try:
            if not self.llm_hotkey:
                print("No LLM hotkey configured")
                return
                
            # Initialize the hotkey manager if it doesn't exist
            if not hasattr(self, 'hotkey_manager') or not self.hotkey_manager:
                self.hotkey_manager = HotkeyManager(self)
                
                # Connect signals
                self.hotkey_manager.hotkey_toggle_signal.connect(self.on_hotkey_toggle)
                self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
                self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
                self.hotkey_manager.error_signal.connect(self.show_error)
                self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)
            
            # Convert the hotkey to the format keyboard library understands
            llm_hotkey_str = self._convert_key_sequence_to_string(QKeySequence(self.llm_hotkey))
            
            # Set mode based on radio buttons
            is_push_to_talk = True  # Default to push-to-talk
            if hasattr(self, 'push_to_talk_radio'):
                is_push_to_talk = self.push_to_talk_radio.isChecked()
            
            # Add or update the LLM hotkey
            self.hotkey_manager.add_hotkey(llm_hotkey_str, "llm", is_push_to_talk)
            
            # Start the manager if it's not already running
            if not self.hotkey_manager.isRunning():
                print(f"Starting hotkey manager with LLM hotkey: {self.llm_hotkey}")
                self.hotkey_manager.start()
            else:
                print(f"Added LLM hotkey {self.llm_hotkey} to existing hotkey manager")
                
        except Exception as e:
            self.show_error(f"Error setting up LLM hotkey: {str(e)}")

    def on_hotkey_press(self, hotkey_id):
        """Handle hotkey press event - starts recording in push-to-talk mode"""
        try:
            print(f"Hotkey press event for ID: {hotkey_id}")
            
            if hotkey_id == "transcription":
                if not self.is_hotkey_recording and not self.is_transcribing:
                    self.start_hotkey_recording()
            elif hotkey_id == "llm":
                if not self.is_recording and not self.is_transcribing:
                    print("Starting LLM recording...")
                    self.start_llm_recording()
        except Exception as e:
            self.show_error(f"Hotkey press error: {str(e)}")

    def on_hotkey_release(self, hotkey_id):
        """Handle hotkey release event - stops recording in push-to-talk mode"""
        try:
            print(f"Hotkey release event for ID: {hotkey_id}")
            
            if hotkey_id == "transcription":
                if self.is_hotkey_recording:
                    self.stop_hotkey_recording()
            elif hotkey_id == "llm":
                if self.is_recording and self.is_llm_recording:
                    print("Stopping LLM recording...")
                    self.stop_recording()  # Call stop_recording directly
                    # Don't reset is_llm_recording flag here, it will be reset after transcription
                    self.show_overlay("Processing recording...")
        except Exception as e:
            self.show_error(f"Hotkey release error: {str(e)}")

    def on_hotkey_toggle(self, hotkey_id):
        """Handle hotkey toggle event - starts or stops recording"""
        try:
            print(f"Hotkey toggle event for ID: {hotkey_id}")
            
            if hotkey_id == "transcription":
                if not self.is_hotkey_recording:
                    # Start recording
                    self.start_hotkey_recording()
                else:
                    # Stop recording
                    self.stop_hotkey_recording()
            elif hotkey_id == "llm":
                if not self.is_recording:
                    self.start_llm_recording()
                else:
                    self.stop_llm_recording()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Hotkey toggle error: {str(e)}\n{error_details}")
            self.show_error(f"Hotkey error: {str(e)}")

    def start_hotkey_recording(self):
        """Start recording when hotkey is pressed"""
        try:
            print("Starting hotkey recording...")
            
            # Check if already recording
            if self.is_hotkey_recording:
                print("Already recording, ignoring start request")
                return
            
            # Start recording
            self.status_bar.showMessage("Hotkey pressed - Recording started")
            
            # Show overlay with recording message
            self.show_overlay("Recording...")
            
            # Use the current audio settings
            device_idx = self.device_combo.currentData()
            sample_rate = int(self.sample_rate_combo.currentText())
            channels = 1 if self.channels_combo.currentIndex() == 0 else 2
            
            print(f"Starting hotkey recording with device: {device_idx}, sample rate: {sample_rate}, channels: {channels}")
            
            # Create and start the recorder
            self.recorder = AudioRecorder(sample_rate, channels, device_idx)
            self.recorder.update_signal.connect(self.update_level_meter)
            self.recorder.finished_signal.connect(self.hotkey_recording_finished)
            self.recorder.error_signal.connect(self.show_error)
            
            self.recorder.start_recording()
            
            # Update recording state
            self.is_hotkey_recording = True
            
            # Start the timer to update the UI
            if hasattr(self, 'hotkey_recording_timer'):
                self.hotkey_recording_timer.start()
            
            # Update the window title to indicate recording
            self.setWindowTitle("Whisper UI - Recording")
            
            print("Recording started successfully")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Start hotkey recording error: {str(e)}\n{error_details}")
            self.show_error(f"Failed to start recording: {str(e)}")
            self.is_hotkey_recording = False
            self.hide_overlay()

    def stop_hotkey_recording(self):
        """Stop recording when hotkey is pressed again or released"""
        print("Stopping hotkey recording...")
        
        if not self.is_hotkey_recording:
            print("Not currently recording, ignoring stop request")
            return
            
        try:
            if self.recorder and self.recorder.recording:
                print("Active recorder found, stopping recording")
                self.status_bar.showMessage("Processing recording...")
                
                # Update overlay
                self.show_overlay("Processing...")
                
                self.recorder.stop_recording()
                
                # Update recording state
                self.is_hotkey_recording = False
                
                # Stop the timer
                if hasattr(self, 'hotkey_recording_timer'):
                    self.hotkey_recording_timer.stop()
                
                # Reset the window title
                self.setWindowTitle("Diktando - Windows dictation tool on steroids")
                
                print("Recording stopped successfully")
                
            else:
                print("No active recorder found, stopping recording")
                self.status_bar.showMessage("Recording stopped")
                self.is_hotkey_recording = False
                self.hide_overlay()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Stop hotkey recording error: {str(e)}\n{error_details}")
            self.show_error(f"Failed to stop recording: {str(e)}")
            self.is_hotkey_recording = False
            self.hide_overlay()

    def handle_transcription_error(self, error_message):
        """Handle transcription errors and save debug recordings"""
        self.log_message(f"Transcription error: {error_message}")
        self.show_overlay(f"ERROR: {error_message}")
        
        try:
            if os.path.exists(self.recording_path):
                # Create a timestamped filename for the debug recording
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                debug_filename = f"failed_transcription_{timestamp}.wav"
                debug_path = os.path.join(self.debug_dir, debug_filename)
                
                # Copy the problematic recording to the debug directory
                import shutil
                shutil.copy2(self.recording_path, debug_path)
                self.log_message(f"Debug recording saved to: {debug_path}")
                
                # Save metadata about the error
                metadata_path = os.path.join(self.debug_dir, f"failed_transcription_{timestamp}.txt")
                with open(metadata_path, 'w') as f:
                    f.write(f"Error: {error_message}\n")
                    f.write(f"Original file: {self.recording_path}\n")
                    f.write(f"File size: {os.path.getsize(self.recording_path)} bytes\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            else:
                self.log_message("Error saving debug recording: Recording file not found")
        except Exception as e:
            self.log_message(f"Error saving debug recording: {str(e)}")

    def hotkey_transcription_finished(self, text):
        """Handle transcription completion from hotkey recording"""
        # Clean up the transcription first
        text = self.clean_transcription(text)
        
        # Set the transcription and attempt to paste it
        self.hotkey_manager.set_transcription(text)
        self.show_overlay("Pasting transcription...")
        self.attempt_paste_transcription()
        
        # Add to history after cleaning
        self.append_to_transcription_history(text, source="hotkey")
        
        # Log the transcription
        self.log_message(f"Transcription completed: {text}")
        
        # Hide overlay after a delay
        QTimer.singleShot(2000, self.hide_overlay)

    def attempt_paste_transcription(self):
        """Attempt to paste the transcription and handle any issues"""
        try:
            # Backup the clipboard before we do anything
            self.hotkey_manager.backup_clipboard()
            
            # Get the content types in the clipboard before we change it
            content_types = self.clipboard_manager.get_clipboard_content_type()
            content_type_str = ", ".join(content_types) if content_types else "empty"
            print(f"Original clipboard content types: {content_type_str}")
            
            # Add a small delay to ensure backup is complete
            time.sleep(0.1)
            
            # Check if auto-paste is enabled
            if self.auto_paste_check.isChecked():
                # Set clipboard content and paste
                self.clipboard_manager.set_clipboard_text(self.hotkey_manager.transcription)
                
                # Try to paste - the result will be handled by the on_paste_complete signal handler
                self.hotkey_manager.paste_transcription()
            else:
                # Just set the clipboard without pasting
                self.clipboard_manager.set_clipboard_text(self.hotkey_manager.transcription)
                self.status_bar.showMessage("Copied to clipboard")
                # Emit the signal manually since we're not calling paste_transcription
                self.hotkey_manager.paste_complete_signal.emit(False)
            
            # Restore the original clipboard after a longer delay to ensure paste completes
            QTimer.singleShot(2000, self.restore_original_clipboard)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error during paste operation: {str(e)}")
            # Restore clipboard immediately if there was an error
            self.restore_original_clipboard()
    
    def on_paste_complete(self, success):
        """Handle completion of paste operation"""
        if success:
            self.show_overlay("Transcription pasted successfully")
            # Hide overlay after a short delay
            QTimer.singleShot(1000, self.hide_overlay)
        else:
            self.show_overlay("Failed to paste transcription")
            # Hide overlay after a longer delay for error messages
            QTimer.singleShot(2000, self.hide_overlay)
        
        # Restore the original clipboard content
        self.restore_original_clipboard()
    
    def restore_original_clipboard(self):
        """Restore the original clipboard content after pasting"""
        try:
            if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                self.hotkey_manager.restore_clipboard()
                print("Original clipboard content restored")
            else:
                print("No hotkey manager available to restore clipboard")
        except Exception as e:
            print(f"Error restoring clipboard: {str(e)}")

    def update_hotkey_recording_indicator(self):
        """Update the UI to indicate that hotkey recording is active"""
        if self.is_hotkey_recording:
            # Change the animation color based on recording state
            if self.animation_container.styleSheet().find("#FF453A") != -1:
                # Switch to a slightly different shade of red for pulsing effect
                self.animation_container.setStyleSheet("""
                    background-color: #FF5A50;
                    border-radius: 8px;
                """)
            else:
                # Switch back to original red
                self.animation_container.setStyleSheet("""
                    background-color: #FF453A;
                    border-radius: 8px;
                """)

    def clean_transcription(self, text, keep_timestamps=False):
        """Remove timestamps and clean up the transcription text"""
        import re
        
        cleaned_text = text
        
        if not keep_timestamps:
        # Remove timestamp lines like [00:00:00.000 --> 00:00:02.580]
            cleaned_text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*', '', cleaned_text)
        
        # Remove [BLANK_AUDIO] markers
        cleaned_text = re.sub(r'\[BLANK_AUDIO\]', '', cleaned_text)
        
        # Remove any extra whitespace
        cleaned_text = re.sub(r'\s+', ' ' if not keep_timestamps else '\n', cleaned_text)
        
        # Trim leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text

    def check_and_download_default_model(self):
        """Check if any models are installed, and if not, download the base.en model"""
        models_dir = self.get_models_dir()
        
        # Check if any models are installed
        model_files = [f for f in os.listdir(models_dir) if f.startswith("ggml-") and f.endswith(".bin")]
        
        if not model_files:
            # No models installed, show a message and download the base.en model
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setWindowTitle("First Run Setup")
            msg.setText("No speech recognition models found.")
            msg.setInformativeText("The application will now download the 'base.en' model (approximately 140MB) to get you started. You can download other models later from the Models tab.")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
            
            # Set the models combo to base.en
            index = self.models_combo.findText("base.en")
            if index >= 0:
                self.models_combo.setCurrentIndex(index)
            
            # Start the download
            self.status_bar.showMessage("Downloading default model: base.en")
            self.show_overlay("Downloading base.en model...")
            
            model_name = "base.en"
            model_url = self.get_model_url(model_name)
            output_dir = models_dir
            
            # Create the downloader in a way that it can be properly cleaned up
            if hasattr(self, 'downloader') and self.downloader and self.downloader.isRunning():
                # If there's already a download running, cancel it first
                print("Canceling existing download before starting new one")
                self.downloader.cancel()
                self.downloader.wait(2000)  # Wait up to 2 seconds for it to finish
            
            # Create a new downloader
            self.downloader = ModelDownloader(model_name, model_url, output_dir)
            self.downloader.progress_signal.connect(self.update_download_progress)
            self.downloader.finished_signal.connect(self.default_model_download_finished)
            self.downloader.error_signal.connect(self.handle_download_error)
            
            # Set the download flag
            self.is_download_in_progress = True
            
            # Start the download in a way that can be properly cleaned up
            print(f"Starting download of {model_name} model")
            self.downloader.start()
            
            # Add a message to inform the user that they can close the application and restart later
            self.status_bar.showMessage(f"Downloading default model: {model_name} (you can close and restart later if needed)")
            
            # Make sure the application doesn't exit immediately
            QApplication.processEvents()
    
    def default_model_download_finished(self):
        """Handle the completion of the default model download"""
        # Reset the download flag
        self.is_download_in_progress = False
        
        self.status_bar.showMessage("Default model downloaded successfully")
        self.hide_overlay()
        self.refresh_models()
        
        # Set the base.en model as the selected model
        index = self.model_combo.findText("base.en")
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        # Show a message to the user
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Setup Complete")
        msg.setText("The base.en model has been downloaded successfully.")
        msg.setInformativeText("You can now start transcribing audio by pressing the F8 key (default hotkey) to start/stop recording. For better quality transcriptions, you may want to download a larger model from the Models tab.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def handle_download_error(self, error_message):
        """Handle download errors"""
        self.is_download_in_progress = False
        self.hide_overlay()
        self.show_error(f"Download failed: {error_message}")
        self.download_progress.setValue(0)
        self.status_bar.showMessage("Download failed")

    def closeEvent(self, event):
        """Handle the window close event"""
        # Clean up resources
        self.cleanup_resources()
        
        # Accept the close event
        event.accept()

    def get_settings_file(self):
        """Get the path to the settings file"""
        settings_dir = self.get_models_dir()  # Reuse the models directory for settings
        return os.path.join(settings_dir, "settings.json")
    
    def load_settings(self):
        """Load application settings"""
        settings_file = self.get_settings_file()
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    
                    # Load hotkey settings with defaults if not found
                    self.hotkey = settings.get('hotkey', 'F8')
                    self.llm_hotkey = settings.get('llm_hotkey', 'F9')
                    
                    # Update UI to reflect loaded hotkeys
                    if hasattr(self, 'hotkey_label'):
                        self.hotkey_label.setText(self.hotkey)
                    
                    if hasattr(self, 'llm_hotkey_label'):
                        self.llm_hotkey_label.setText(self.llm_hotkey)
                    
                    # Load dark mode setting
                    dark_mode = settings.get('dark_mode', False)
                    self.dark_mode_check.setChecked(dark_mode)
                    self.toggle_dark_mode(dark_mode)
                    
                    # Load auto-check updates setting
                    auto_check = settings.get('auto_check_updates', True)
                    self.auto_check_updates.setChecked(auto_check)
                    
                    # Load other settings as needed
            else:
                self._set_default_settings()
        except Exception as e:
            print(f"Error loading settings: {str(e)}")
            self._set_default_settings()
    
    def _set_default_settings(self):
        """Set default application settings"""
        # Set default hotkeys
        self.hotkey = 'F8'
        self.llm_hotkey = 'F9'
        
        # Update UI to reflect default hotkeys
        if hasattr(self, 'hotkey_label'):
            self.hotkey_label.setText(self.hotkey)
        
        if hasattr(self, 'llm_hotkey_label'):
            self.llm_hotkey_label.setText(self.llm_hotkey)
        
        # Set default mode
        if hasattr(self, 'push_to_talk_radio'):
            self.push_to_talk_radio.setChecked(True)
            self.toggle_mode_radio.setChecked(False)
        
        # Set default dark mode
        if hasattr(self, 'dark_mode_check'):
            self.dark_mode_check.setChecked(True)
            self.toggle_dark_mode(True)
        
        # Set default auto-check updates
        if hasattr(self, 'auto_check_updates'):
            self.auto_check_updates.setChecked(True)
        
        # Initialize hotkey managers with default settings
        self.toggle_hotkey(True)  # This will create and configure the hotkey manager
        self.setup_llm_hotkey()   # This will create and configure the LLM hotkey manager
    
    def save_settings(self):
        """Save application settings"""
        settings_file = self.get_settings_file()
        try:
            settings = {
                'hotkey': self.hotkey,
                'llm_hotkey': self.llm_hotkey,
                'push_to_talk': self.push_to_talk_radio.isChecked() if hasattr(self, 'push_to_talk_radio') else True,
                'dark_mode': self.dark_mode_check.isChecked() if hasattr(self, 'dark_mode_check') else True,
                'auto_check_updates': self.auto_check_updates.isChecked() if hasattr(self, 'auto_check_updates') else True
            }
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            with open(settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving settings: {str(e)}")

    def on_mode_changed(self):
        """Handle changes in the hotkey mode"""
        # Only process if hotkey is enabled
        if not self.enable_hotkey_check.isChecked():
            return
            
        # Save the settings
        self.save_settings()
        
        # Update the mode for both hotkeys in the unified manager
        if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
            is_push_to_talk = self.push_to_talk_radio.isChecked()
            
            # Update mode for transcription hotkey
            self.hotkey_manager.set_mode("transcription", is_push_to_talk)
            
            # Update mode for LLM hotkey
            self.hotkey_manager.set_mode("llm", is_push_to_talk)
        
        # Update status bar
        mode_str = "Push-to-Talk" if self.push_to_talk_radio.isChecked() else "Toggle"
        self.status_bar.showMessage(f"Switched to {mode_str} mode")
        
        # Update hotkey info text based on mode
        if self.push_to_talk_radio.isChecked():
            hotkey_info_text = (
                "Hold down the hotkey while speaking, release when done.\n"
                "The transcription will automatically replace your clipboard content\n"
                "and be pasted at the cursor position (if auto-paste is enabled).\n\n"
                "When recording is active, the window title will show 'RECORDING'.\n"
                "If you experience transcription errors, try speaking more clearly and\n"
                "ensure your microphone is working properly."
            )
        else:
            hotkey_info_text = (
                "Press the hotkey once to start recording, press again to stop and transcribe.\n"
                "The transcription will automatically replace your clipboard content\n"
                "and be pasted at the cursor position (if auto-paste is enabled).\n\n"
                "When recording is active, the window title will show 'RECORDING'.\n"
                "If you experience transcription errors, try speaking more clearly and\n"
                "ensure your microphone is working properly."
            )
        
        # Find and update the hotkey info label
        for child in self.findChildren(QLabel):
            if hasattr(child, 'text') and child.text() and "press the hotkey" in child.text().lower():
                child.setText(hotkey_info_text)
                break

    def hotkey_recording_finished(self, file_path):
        """Handle finished recording from hotkey"""
        self.log_message(f"Hotkey recording finished, processing file: {file_path}")
        
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                self.log_message(f"Audio file size: {file_size} bytes")
                
                # Check if we need to pad the audio
                was_padded = self.pad_audio_if_needed(file_path)
                if was_padded:
                    self.log_message("Audio was padded with silence")
                
                # Get models directory and set up transcription
                models_dir = self.get_models_dir()
                model_path = os.path.join(models_dir, "ggml-base.en.bin")
                
                # Create and start transcriber
                self.transcriber = Transcriber(file_path, model_path, "en")
                
                # Connect to different handlers based on whether this is LLM or normal transcription
                if hasattr(self, 'is_llm_recording') and self.is_llm_recording:
                    self.log_message("Connecting to LLM transcription handler (is_llm_recording=True)")
                    self.transcriber.finished_signal.connect(self.llm_transcription_finished)
                else:
                    self.log_message("Connecting to regular transcription handler (is_llm_recording=False)")
                    self.transcriber.finished_signal.connect(self.hotkey_transcription_finished)
                
                self.transcriber.error_signal.connect(self.handle_transcription_error)
                self.transcriber.start()
                
                self.show_overlay("Transcribing audio...")
            else:
                self.log_message("Error: Recording file not found")
                self.show_overlay("ERROR: Recording file not found")
        except Exception as e:
            self.log_message(f"Error processing recording: {str(e)}")
            self.show_overlay("ERROR: Failed to process recording")
    
    def llm_transcription_finished(self, text):
        """Handle transcription completion specifically for LLM processing"""
        try:
            # Clean up the transcription first
            cleaned_text = self.clean_transcription(text)
            
            # Add to history
            self.append_to_transcription_history(cleaned_text, source="llm-input")
            
            # Start LLM processing
            self.show_overlay("Processing with LLM...")
            self.start_llm_processing(cleaned_text)
            
            # Log the transcription
            self.log_message(f"LLM transcription completed: {cleaned_text}")
            
            # Reset the LLM recording flag
            self.is_llm_recording = False
            
        except Exception as e:
            self.show_error(f"Error in LLM transcription: {str(e)}")
            self.hide_overlay()
            # Reset the LLM recording flag even if there's an error
            self.is_llm_recording = False

    def manual_trigger_release(self):
        """Manually trigger the hotkey release event for debugging"""
        print("Manually triggering hotkey release event")
        if self.is_hotkey_recording:
            print("Recording is active, manually triggering release event")
            
            # Try to force release via the hotkey manager first
            if hasattr(self, 'hotkey_manager') and self.hotkey_manager and self.hotkey_manager.isRunning():
                if self.hotkey_manager.force_release("transcription"):
                    print("Force release successful via hotkey manager")
                    return
            
            # If that didn't work, call the release handler directly
            print("Calling on_hotkey_release directly")
            self.on_hotkey_release("transcription")
        elif self.is_llm_recording:
            print("LLM recording is active, manually triggering release event")
            
            # Try to force release via the hotkey manager
            if hasattr(self, 'hotkey_manager') and self.hotkey_manager and self.hotkey_manager.isRunning():
                if self.hotkey_manager.force_release("llm"):
                    print("Force release successful via hotkey manager")
                    return
            
            # If that didn't work, call the release handler directly
            print("Calling on_hotkey_release directly")
            self.on_hotkey_release("llm")
        else:
            print("Not currently recording, nothing to release")
            self.status_bar.showMessage("Not recording - nothing to release")

    def log_message(self, message):
        """Add a message to the debug log"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Use the logging system if in debug mode
        if DEBUG:
            logger.info(message)
        else:
            # Fallback to print for non-debug mode
            print(f"[{timestamp}] {message}")
        
        # If we have a debug log text widget, add the message there too
        if hasattr(self, 'debug_log') and self.debug_log is not None:
            self.debug_log.append(f"[{timestamp}] {message}")
            # Scroll to the bottom
            self.debug_log.verticalScrollBar().setValue(
                self.debug_log.verticalScrollBar().maximum()
            )

    def generate_silence_file(self):
        """Generate a 1-second silence WAV file using ffmpeg"""
        try:
            # Get user's app data directory for binaries
            app_data_dir = os.path.join(os.getenv('APPDATA') if platform.system() == "Windows" 
                                      else os.path.expanduser('~/.config'), 'Diktando')
            bin_dir = os.path.join(app_data_dir, 'bin')
            ffmpeg_exe = os.path.join(bin_dir, "ffmpeg.exe") if platform.system() == "Windows" else "ffmpeg"
            
            cmd = [
                ffmpeg_exe, "-f", "lavfi",
                "-i", "anullsrc=r=16000:cl=mono",
                "-t", "1",
                "-acodec", "pcm_s16le",
                "-y",
                self.silence_path
            ]
            
            self.log_message("Generating silence file...")
            subprocess.run(cmd, check=True, creationflags=SUBPROCESS_FLAGS)
            self.log_message(f"Silence file generated at: {self.silence_path}")
        except Exception as e:
            self.log_message(f"Error generating silence file: {str(e)}")
            self.show_error("Failed to generate silence file")

    def pad_audio_if_needed(self, audio_path):
        """Pad audio with silence if it's shorter than 1 second"""
        try:
            # Read the audio file
            audio_data, sample_rate = sf.read(audio_path)
            duration = len(audio_data) / sample_rate
            
            # Log the original duration
            self.log_message(f"Recording duration: {duration:.3f}s")
            
            # If duration is less than 1.1 seconds (giving some buffer), pad it
            if duration < 1.1:
                self.log_message(f"Recording duration: {duration:.3f}s - Adding silence padding")
                
                # Read the silence file
                silence_data, _ = sf.read(self.silence_path)
                
                # Calculate how much silence we need (add extra 0.1s for safety)
                silence_needed = int((1.1 - duration) * sample_rate)
                silence_samples = silence_data[:silence_needed]
                
                # Concatenate the recording with silence
                padded_audio = np.concatenate([audio_data, silence_samples])
                
                # Save back to the same file
                sf.write(audio_path, padded_audio, sample_rate)
                
                # Verify the new duration
                new_duration = len(padded_audio) / sample_rate
                self.log_message(f"Padded audio duration: {new_duration:.3f}s")
                self.log_message("Audio was padded with silence")
                
                return True
            return False
            
        except Exception as e:
            self.log_message(f"Error padding audio: {str(e)}")
            return False

    def check_required_binaries(self):
        """Check and download required binary files"""
        try:
            # Get user's app data directory
            app_data_dir = os.path.join(os.getenv('APPDATA') if platform.system() == "Windows" 
                                      else os.path.expanduser('~/.config'), 'Diktando')
            bin_dir = os.path.join(app_data_dir, 'bin')
            os.makedirs(bin_dir, exist_ok=True)
            
            # Base URL for all files
            base_url = "https://github.com/fspecii/whisper-cpp-start/raw/main/"
            
            # List of required files
            required_files = [
                "ffmpeg.exe",
                "ggml-base.dll",
                "ggml-cpu.dll",
                "ggml.dll",
                "whisper-cpp.exe",
                "whisper.dll"
            ]
            
            # Check if all files exist
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(bin_dir, f))]
            
            if not missing_files:
                print("[{}] All required files are present.".format(time.strftime("%H:%M:%S")))
                return True
                
            print("[{}] Downloading missing binary files...".format(time.strftime("%H:%M:%S")))
            
            # Show initial message with yellow dot for in-progress state
            self.show_overlay("Downloading required files...")
            self.animation_container.setStyleSheet("""
                background-color: #FF9F0A;
                border-radius: 8px;
            """)
            self.status_bar.showMessage("Downloading required binary files...")
            
            # Start the overlay animation
            self.overlay_animation_timer.start()
            
            # Configure requests session with retries
            session = requests.Session()
            retries = requests.adapters.Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[408, 429, 500, 502, 503, 504]
            )
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
            
            # Download missing files
            total_files = len(missing_files)
            for idx, filename in enumerate(missing_files, 1):
                file_path = os.path.join(bin_dir, filename)
                
                try:
                    progress_msg = f"Downloading {filename}... ({idx}/{total_files})"
                    print("[{}] {}".format(time.strftime("%H:%M:%S"), progress_msg))
                    
                    # Update overlay with current file progress
                    self.show_overlay(progress_msg)
                    self.animation_container.setStyleSheet("""
                        background-color: #FF9F0A;
                        border-radius: 8px;
                    """)
                    self.status_bar.showMessage(progress_msg)
                    
                    # Process events to ensure UI updates
                    QApplication.processEvents()
                    
                    # Download with timeout
                    url = base_url + filename
                    response = session.get(url, timeout=(10, 30))
                    response.raise_for_status()
                    
                    # Save the file
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    success_msg = f"Successfully downloaded {filename}"
                    print("[{}] {}".format(time.strftime("%H:%M:%S"), success_msg))
                    self.status_bar.showMessage(success_msg)
                    
                except Exception as e:
                    error_msg = f"Failed to download {filename}: {str(e)}"
                    print("[{}] Error: {}".format(time.strftime("%H:%M:%S"), error_msg))
                    self.show_error(error_msg)
                    self.overlay_animation_timer.stop()
                    return False
            
            # Show completion message with green dot
            completion_msg = "Required files downloaded successfully"
            self.show_overlay(completion_msg)
            self.animation_container.setStyleSheet("""
                background-color: #30D158;
                border-radius: 8px;
            """)
            self.status_bar.showMessage(completion_msg)
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # Hide the overlay after a short delay
            QTimer.singleShot(2000, lambda: (self.hide_overlay(), self.overlay_animation_timer.stop()))
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to download required files: {str(e)}"
            print("[{}] Error: {}".format(time.strftime("%H:%M:%S"), error_msg))
            self.show_error(error_msg)
            self.overlay_animation_timer.stop()
            return False

    def append_to_transcription_history(self, text, source="file"):
        """Add a transcription to the history with timestamp and source info"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Add timestamp and source info
        history_entry = f"[{timestamp}] - {source.upper()}\n{text}\n{'-' * 80}\n\n"
        
        # Add to history list
        self.transcription_history.append(history_entry)
        
        # Update the results text area with all history
        self.results_text.setPlainText("".join(self.transcription_history))
        
        # Scroll to the bottom to show the latest transcription
        self.results_text.moveCursor(QTextCursor.End)
        
        # Return the original text - cleaning is now done before calling this method
        return text

    def clear_transcription_history(self):
        """Clear the transcription history"""
        self.transcription_history = []
        self.results_text.clear()
        self.status_bar.showMessage("Transcription history cleared")

    def export_as_text(self):
        """Export transcription as a text file"""
        try:
            # Create a dialog for export options
            dialog = QDialog(self)
            dialog.setWindowTitle("Export Options")
            layout = QVBoxLayout(dialog)
            
            # Add timestamp options group
            timestamp_group = QGroupBox("Timestamp Options")
            timestamp_layout = QVBoxLayout(timestamp_group)
            
            # Add radio buttons for timestamp options
            no_timestamps = QRadioButton("No timestamps (clean text only)")
            no_timestamps.setChecked(True)
            timestamp_layout.addWidget(no_timestamps)
            
            history_timestamps = QRadioButton("Include history timestamps (date/time)")
            timestamp_layout.addWidget(history_timestamps)
            
            speech_timestamps = QRadioButton("Include speech timestamps (time ranges)")
            timestamp_layout.addWidget(speech_timestamps)
            
            timestamp_group.setLayout(timestamp_layout)
            layout.addWidget(timestamp_group)
            
            # Add buttons
            button_box = QHBoxLayout()
            ok_button = QPushButton("OK", dialog)
            ok_button.clicked.connect(dialog.accept)
            cancel_button = QPushButton("Cancel", dialog)
            cancel_button.clicked.connect(dialog.reject)
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export as Text", "", 
                    "Text Files (*.txt);;All Files (*)"
                )
                
                if file_path:
                    text = self.results_text.toPlainText()
                    
                    # Process text based on selected option
                    if no_timestamps.isChecked():
                        # Remove all timestamps and clean the text
                        output_text = self.clean_transcription(text, keep_timestamps=False)
                    elif history_timestamps.isChecked():
                        # Keep only history timestamps
                        lines = text.split('\n')
                        output_lines = []
                        for line in lines:
                            if line.startswith('[20'):  # History timestamp
                                output_lines.append(line)
                            elif not line.startswith('['):  # Not a timestamp line
                                output_lines.append(line)
                        output_text = '\n'.join(output_lines)
                    else:  # speech_timestamps.isChecked()
                        # Keep only speech timestamps
                        lines = text.split('\n')
                        output_lines = []
                        for line in lines:
                            if line.startswith('[') and '-->' in line:  # Speech timestamp
                                output_lines.append(line)
                            elif not line.startswith('['):  # Not a timestamp line
                                output_lines.append(line)
                        output_text = '\n'.join(output_lines)
                    
                    # Write the processed text to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(output_text)
                    
                    self.status_bar.showMessage(f"Successfully exported to {file_path}")
                    self.show_overlay("Text file exported successfully")
                    QTimer.singleShot(2000, self.hide_overlay)
    
        except Exception as e:
            self.show_error(f"Failed to export as text: {str(e)}")

    def export_as_pdf(self):
        """Export transcription as a PDF file"""
        try:
            # Create a dialog for export options
            dialog = QDialog(self)
            dialog.setWindowTitle("Export Options")
            layout = QVBoxLayout(dialog)
            
            # Add timestamp options group
            timestamp_group = QGroupBox("Timestamp Options")
            timestamp_layout = QVBoxLayout(timestamp_group)
            
            # Add radio buttons for timestamp options
            no_timestamps = QRadioButton("No timestamps (clean text only)")
            no_timestamps.setChecked(True)
            timestamp_layout.addWidget(no_timestamps)
            
            history_timestamps = QRadioButton("Include history timestamps (date/time)")
            timestamp_layout.addWidget(history_timestamps)
            
            speech_timestamps = QRadioButton("Include speech timestamps (time ranges)")
            timestamp_layout.addWidget(speech_timestamps)
            
            timestamp_group.setLayout(timestamp_layout)
            layout.addWidget(timestamp_group)
            
            # Add buttons
            button_box = QHBoxLayout()
            ok_button = QPushButton("OK", dialog)
            ok_button.clicked.connect(dialog.accept)
            cancel_button = QPushButton("Cancel", dialog)
            cancel_button.clicked.connect(dialog.reject)
            button_box.addWidget(ok_button)
            button_box.addWidget(cancel_button)
            layout.addLayout(button_box)
            
            if dialog.exec() == QDialog.DialogCode.Accepted:
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "Export as PDF", "", 
                    "PDF Files (*.pdf);;All Files (*)"
                )
                
                if file_path:
                    from PyQt5.QtPrintSupport import QPrinter
                    from PyQt5.QtGui import QTextDocument
                    
                    text = self.results_text.toPlainText()
                    
                    # Process text based on selected option
                    if no_timestamps.isChecked():
                        # Remove all timestamps and clean the text
                        output_text = self.clean_transcription(text, keep_timestamps=False)
                    elif history_timestamps.isChecked():
                        # Keep only history timestamps
                        lines = text.split('\n')
                        output_lines = []
                        for line in lines:
                            if line.startswith('[20'):  # History timestamp
                                output_lines.append(line)
                            elif not line.startswith('['):  # Not a timestamp line
                                output_lines.append(line)
                        output_text = '\n'.join(output_lines)
                    else:  # speech_timestamps.isChecked()
                        # Keep only speech timestamps
                        lines = text.split('\n')
                        output_lines = []
                        for line in lines:
                            if line.startswith('[') and '-->' in line:  # Speech timestamp
                                output_lines.append(line)
                            elif not line.startswith('['):  # Not a timestamp line
                                output_lines.append(line)
                        output_text = '\n'.join(output_lines)
                    
                    # Create PDF
                    printer = QPrinter()
                    printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                    printer.setOutputFileName(file_path)
                    
                    doc = QTextDocument()
                    doc.setPlainText(output_text)
                    doc.print_(printer)
                    
                    self.status_bar.showMessage(f"Successfully exported to {file_path}")
                    self.show_overlay("PDF file exported successfully")
                    QTimer.singleShot(2000, self.hide_overlay)
    
        except Exception as e:
            self.show_error(f"Failed to export as PDF: {str(e)}")

    def export_as_srt(self):
        """Export transcription as an SRT file with timestamps"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export as SRT", "", 
                "SubRip Files (*.srt);;All Files (*)"
            )
            
            if file_path:
                # Get the text and clean it to ensure we only have speech timestamps
                text = self.results_text.toPlainText()
                
                # Remove history timestamps (lines starting with [YYYY-MM-DD])
                lines = text.split('\n')
                speech_lines = []
                for line in lines:
                    # Skip history timestamp lines and separator lines
                    if not line.strip().startswith('[20') and not line.strip().startswith('-' * 10):
                        speech_lines.append(line)
                
                # Rejoin the lines and convert to SRT
                speech_text = '\n'.join(speech_lines)
                self.log_message("Processing text for SRT conversion after removing history timestamps")
                
                # Convert to SRT format
                srt_content = self._convert_to_srt(speech_text)
                
                # Verify we have content before writing
                if not srt_content.strip():
                    raise ValueError("No valid speech timestamps found in the transcription")
                
                # Write the SRT file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                
                self.status_bar.showMessage(f"Successfully exported to {file_path}")
                self.show_overlay("SRT file exported successfully")
                QTimer.singleShot(2000, self.hide_overlay)
    
        except Exception as e:
            self.log_message(f"Error during SRT export: {str(e)}")
            self.show_error(f"Failed to export as SRT: {str(e)}")

    def _convert_to_srt(self, text):
        """Convert transcription text with timestamps to SRT format"""
        import re
        
        # Initialize variables
        srt_entries = []
        counter = 1
        
        # Log the input text for debugging
        self.log_message(f"Converting to SRT format. Input text length: {len(text)}")
        self.log_message("First 200 characters of input:")
        self.log_message(text[:200])
        
        try:
            # Regular expression to match timestamp lines and associated text
            # This pattern looks for [HH:MM:SS.mmm --> HH:MM:SS.mmm] followed by text
            pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*?)(?=\s*\[|\s*$)'
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            
            for match in matches:
                start_time = match.group(1)
                end_time = match.group(2)
                text_content = match.group(3).strip()
                
                # Skip if text is empty or contains only [BLANK_AUDIO]
                if not text_content or text_content == '[BLANK_AUDIO]':
                    continue
                
                # Log the found timestamp and text
                self.log_message(f"Found timestamp: {start_time} --> {end_time}")
                self.log_message(f"Text content: {text_content[:50]}...")
                
                # Convert timestamps from [HH:MM:SS.mmm] to SRT format (HH:MM:SS,mmm)
                start_srt = start_time.replace('.', ',')
                end_srt = end_time.replace('.', ',')
                
                # Format SRT entry
                srt_entry = f"{counter}\n{start_srt} --> {end_srt}\n{text_content}\n\n"
                srt_entries.append(srt_entry)
                counter += 1
            
            if not srt_entries:
                self.log_message("No valid timestamps found in the text")
                raise ValueError("No valid speech timestamps found in the transcription")
            
            self.log_message(f"Successfully created {len(srt_entries)} SRT entries")
            return "".join(srt_entries)
            
        except Exception as e:
            self.log_message(f"Error in _convert_to_srt: {str(e)}")
            self.log_message("Full text being processed:")
            self.log_message(text)
            raise

    def show_update_dialog(self, current_version, latest_version):
        """Show the update dialog"""
        # Prevent multiple update dialogs
        if hasattr(self, 'update_dialog') and self.update_dialog.isVisible():
            self.update_dialog.activateWindow()
            return

        # Create and show the update dialog
        self.update_dialog = UpdateDialog(self, current_version, latest_version)
        
        # Connect signals
        self.update_checker.update_progress_signal.connect(self.update_dialog.update_progress)
        self.update_checker.update_success_signal.connect(self.handle_update_success)
        self.update_checker.update_error_signal.connect(self.handle_update_error)
        
        # Show the dialog
        self.update_dialog.show()

    def handle_update_success(self):
        """Handle successful update"""
        if hasattr(self, 'update_dialog'):
            self.update_dialog.close()
        
        # Show a message about the successful update
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Update Successful")
        msg.setText("The application has been updated successfully.")
        msg.setInformativeText("The application will now restart to complete the update.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()
        
        # Clean up resources before exiting
        self.cleanup_resources()
        
        # Exit the application - the batch script will restart it
        QApplication.quit()

    def handle_update_error(self, error_message):
        """Handle update error"""
        if hasattr(self, 'update_dialog'):
            self.update_dialog.close()
        
        # Show error message
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Update Failed")
        msg.setText("Failed to update the application.")
        msg.setInformativeText(f"Error: {error_message}\n\nPlease try again later or download the latest version manually.")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _cleanup_update_dialog(self):
        """Clean up update dialog connections"""
        if hasattr(self, 'update_dialog'):
            self.update_checker.update_progress_signal.disconnect(self.update_dialog.update_progress)
            self.update_checker.update_success_signal.disconnect(self.handle_update_success)
            self.update_checker.update_error_signal.disconnect(self.handle_update_error)
            delattr(self, 'update_dialog')

    def update_progress(self, progress):
        """Update the progress of the download"""
        self.status_bar.showMessage(f"Downloading update: {progress}%")
        self.show_overlay(f"Downloading: {progress}%")
    
    def handle_update_success(self):
        """Handle successful update download"""
        self.status_bar.showMessage("Update downloaded. Restarting application...")
        self.show_overlay("Update successful! Restarting...")
        QTimer.singleShot(2000, self.close)  # Close after 2 seconds

    def show_llm_settings(self, show_hotkey_tab=False):
        """Show LLM settings dialog"""
        dialog = LLMSettingsDialog(self, self.llm_processor, self.llm_hotkey)
        
        # If we're showing the hotkey tab, focus on that
        if show_hotkey_tab and hasattr(dialog, 'hotkey_edit'):
            dialog.hotkey_edit.setFocus()
        
        if dialog.exec_():
            settings = dialog.get_settings()
            
            # Update LLM processor settings
            self.llm_processor.set_provider(settings['provider'], settings['api_key'], settings['model'])
            self.llm_processor.set_prompt_template(settings['prompt_template'])
            self.llm_processor.set_mode(settings['is_push_to_talk'])
            self.llm_processor.set_include_screenshot(settings['include_screenshot'])
            
            # Update hotkey if changed
            new_hotkey = settings['hotkey']
            if new_hotkey != self.llm_hotkey:
                self.llm_hotkey = new_hotkey
                
                # Update UI to reflect the new hotkey
                if hasattr(self, 'llm_hotkey_label'):
                    self.llm_hotkey_label.setText(self.llm_hotkey if self.llm_hotkey else "F9")
                
                # Update the hotkey in the unified manager
                if self.llm_hotkey and hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                    # Convert to keyboard library format
                    llm_hotkey_str = self._convert_key_sequence_to_string(QKeySequence(self.llm_hotkey))
                    
                    # Update the hotkey
                    is_push_to_talk = self.push_to_talk_radio.isChecked()
                    
                    # Add or update the LLM hotkey
                    self.hotkey_manager.add_hotkey(llm_hotkey_str, "llm", is_push_to_talk)
                    
                    # Restart the manager if it's running
                    if self.hotkey_manager.isRunning():
                        self.hotkey_manager.stop()
                        self.hotkey_manager.wait()
                        self.hotkey_manager.start()
                        
                    print(f"Updated LLM hotkey to: {self.llm_hotkey}")
                elif not self.llm_hotkey and hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                    # Remove the LLM hotkey
                    self.hotkey_manager.remove_hotkey("llm")
                    print("Removed LLM hotkey")
            
            # Save all settings
            self.save_settings()

        if show_hotkey_tab:
            self.hotkey_label.setText(self.llm_hotkey)
            self.llm_hotkey_label.setText(self.llm_hotkey)

    def start_llm_recording(self):
        """Start recording for LLM processing"""
        try:
            # Set flags to indicate this is for LLM
            self.is_llm_recording = True
            
            # Start recording using existing method
            self.start_recording()
            self.show_overlay("Recording for LLM...")
        except Exception as e:
            self.is_llm_recording = False
            self.show_error(f"Error starting LLM recording: {str(e)}")
            self.hide_overlay()

    def stop_llm_recording(self):
        """Stop recording and initiate LLM processing chain"""
        try:
            if self.recorder and self.recorder.recording:
                print("Stopping LLM recording...")
                self.stop_recording()  # This will handle the recorder cleanup
                self.is_llm_recording = False
                self.show_overlay("Transcribing for LLM...")
        except Exception as e:
            self.is_llm_recording = False
            self.show_error(f"Error stopping LLM recording: {str(e)}")
            self.hide_overlay()

    def start_llm_processing(self, text):
        """Start LLM processing with provided text"""
        try:
            if text:
                self.log_message(f"Starting LLM processing with text: {text[:100]}...")
                
                # Check if LLM processor is initialized
                if not hasattr(self, 'llm_processor') or not self.llm_processor:
                    self.log_message("Error: LLM processor not initialized")
                    self.show_error("LLM processor not initialized")
                    return
                
                # Check if API key is configured
                if not self.llm_processor.api_key:
                    self.log_message("Error: No API key configured for LLM")
                    self.show_error("No API key configured. Please set up LLM settings first.")
                    return
                
                # Log screenshot inclusion status
                if self.llm_processor.include_screenshot:
                    self.log_message("Screenshot will be included with LLM request")
                    self.show_overlay("Processing with LLM (with screenshot)...")
                else:
                    self.show_overlay("Processing with LLM...")
                
                # Start processing
                self.llm_processor.start_processing(text)
            else:
                self.log_message("Error: No text to process with LLM")
                self.show_error("No text to process")
        except Exception as e:
            self.log_message(f"Error starting LLM processing: {str(e)}")
            self.show_error(f"Error starting LLM processing: {str(e)}")
            self.hide_overlay()

    def stop_llm_processing(self):
        """Stop LLM processing"""
        try:
            self.llm_processor.stop_processing()
            self.hide_overlay()
        except Exception as e:
            self.show_error(f"Error stopping LLM processing: {str(e)}")
            self.hide_overlay()

    def on_llm_processing_complete(self, processed_text):
        """Handle LLM processing completion"""
        try:
            if processed_text:
                # Log whether screenshot was included in the request
                if hasattr(self, 'llm_processor') and self.llm_processor.include_screenshot:
                    self.log_message(f"LLM processing complete with screenshot. Result: {processed_text[:100]}...")
                else:
                    self.log_message(f"LLM processing complete. Result: {processed_text[:100]}...")
                
                # Add to history with LLM response tag
                self.append_to_transcription_history(processed_text, source="llm-response")
                
                # Set the processed text for pasting using the hotkey manager
                if hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                    # Show overlay and attempt to paste
                    self.show_overlay("Pasting LLM response...")
                    
                    # Backup clipboard before pasting
                    self.log_message("Backing up clipboard content")
                    self.hotkey_manager.backup_clipboard()
                    
                    # Add a small delay to ensure backup is complete
                    time.sleep(0.1)
                    
                    # Set the transcription text in the hotkey manager
                    self.hotkey_manager.set_transcription(processed_text)
                    
                    # Set clipboard content and paste
                    if self.auto_paste_check.isChecked():
                        self.log_message("Auto-paste enabled, pasting LLM response")
                        paste_success = self.hotkey_manager.paste_transcription()
                        if paste_success:
                            self.log_message("LLM response pasted successfully")
                        else:
                            self.log_message("Failed to paste LLM response")
                    else:
                        # Just set the clipboard without pasting
                        self.log_message("Auto-paste disabled, copying LLM response to clipboard")
                        self.clipboard_manager.set_clipboard_text(processed_text)
                        self.status_bar.showMessage("LLM response copied to clipboard")
                        self.hotkey_manager.paste_complete_signal.emit(False)
                    
                    # Restore the original clipboard after a delay
                    QTimer.singleShot(2000, self.restore_original_clipboard)
                else:
                    self.log_message("Error: Hotkey manager not initialized")
                    self.show_error("Hotkey manager not initialized")
            else:
                self.log_message("Error: LLM processing returned empty result")
                self.show_error("LLM processing returned empty result")
        except Exception as e:
            self.log_message(f"Error handling LLM result: {str(e)}")
            self.show_error(f"Error handling LLM result: {str(e)}")
            import traceback
            self.log_message(traceback.format_exc())
        finally:
            QTimer.singleShot(3000, self.hide_overlay)  # Hide overlay after 3 seconds

    def show_hotkey_config(self):
        """Show the hotkey configuration dialog"""
        current_hotkey = self.hotkey_label.text()
        dialog = HotkeyConfigDialog(self, current_hotkey)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_hotkey = dialog.get_hotkey()
            if new_hotkey:
                # Update the label
                self.hotkey_label.setText(new_hotkey)
                self.hotkey = new_hotkey
                
                # Save the settings
                self.save_settings()
                
                # If hotkey is enabled, update the hotkey manager
                if self.enable_hotkey_check.isChecked() and hasattr(self, 'hotkey_manager') and self.hotkey_manager:
                    # Convert to keyboard library format
                    hotkey_str = self._convert_key_sequence_to_string(QKeySequence(new_hotkey))
                    
                    # Update the hotkey
                    is_push_to_talk = self.push_to_talk_radio.isChecked()
                    
                    # Stop the manager if it's running
                    if self.hotkey_manager.isRunning():
                        self.hotkey_manager.stop()
                        self.hotkey_manager.wait()
                    
                    # Add or update the transcription hotkey
                    self.hotkey_manager.add_hotkey(hotkey_str, "transcription", is_push_to_talk)
                    
                    # Add LLM hotkey if configured
                    if hasattr(self, 'llm_hotkey') and self.llm_hotkey:
                        llm_hotkey_str = self._convert_key_sequence_to_string(QKeySequence(self.llm_hotkey))
                        self.hotkey_manager.add_hotkey(llm_hotkey_str, "llm", is_push_to_talk)
                    
                    # Start the manager
                    self.hotkey_manager.start()
                
                self.status_bar.showMessage(f"Hotkey updated to: {new_hotkey}")
            else:
                self.show_error("Please enter a valid hotkey")
    
    def _convert_key_sequence_to_string(self, key_sequence):
        """Convert a QKeySequence to a string format for the keyboard library"""
        key_text = key_sequence.toString()
        
        # Map Qt key names to keyboard library key names
        key_map = {
            "F1": "f1", "F2": "f2", "F3": "f3", "F4": "f4",
            "F5": "f5", "F6": "f6", "F7": "f7", "F8": "f8",
            "F9": "f9", "F10": "f10", "F11": "f11", "F12": "f12",
            "Space": "space", "Return": "enter", "Tab": "tab",
            "Escape": "esc", "Ctrl": "ctrl", "Alt": "alt",
            "Shift": "shift", "Meta": "windows"
        }
        
        # Handle key combinations
        if "+" in key_text:
            parts = key_text.split("+")
            # Convert each part using the map
            converted_parts = []
            for part in parts:
                part = part.strip()
                # Convert the key using the map, or use lowercase if not in map
                converted_parts.append(key_map.get(part, part.lower()))
            # Join with + to create the hotkey string
            return "+".join(converted_parts)
        else:
            # Single key
            return key_map.get(key_text, key_text.lower())


class UpdateDialog(QDialog):
    """Dialog for updating the application"""
    
    def __init__(self, parent=None, current_version=None, latest_version=None):
        super().__init__(parent)
        self.setWindowTitle("Update Available")
        self.setModal(True)
        self.setMinimumWidth(400)
        
        self.current_version = current_version
        self.latest_version = latest_version
        self.update_in_progress = False
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add icon and title
        header_layout = QHBoxLayout()
        
        # Add update icon
        update_icon = QLabel()
        update_icon.setPixmap(QIcon.fromTheme("system-software-update").pixmap(48, 48))
        header_layout.addWidget(update_icon)
        
        # Add title and description
        title_layout = QVBoxLayout()
        title = QLabel(f"<h3>Update Available: {latest_version}</h3>")
        description = QLabel(f"You are currently using version {current_version}. Would you like to update to the latest version?")
        description.setWordWrap(True)
        
        title_layout.addWidget(title)
        title_layout.addWidget(description)
        
        header_layout.addLayout(title_layout)
        layout.addLayout(header_layout)
        
        # Add progress bar (hidden initially)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Add status label (hidden initially)
        self.status_label = QLabel("Downloading update...")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.update_button = QPushButton("Update Now", self)
        self.update_button.clicked.connect(self._handle_update_click)
        
        self.cancel_button = QPushButton("Later", self)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.update_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def _handle_update_click(self):
        """Handle the update button click"""
        self.update_in_progress = True
        self.show_progress()
        
        # Get the latest release info
        latest_release = self.parent().update_checker.check_for_updates(silent=True)
        if latest_release:
            # Start the update process
            self.parent().update_checker.download_and_install_update(latest_release)
        else:
            self.parent().handle_update_error("Failed to get latest release information")
    
    def show_progress(self):
        """Show the progress bar and update the dialog"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setVisible(True)
        self.status_label.setText("Downloading update...")
        
        # Disable buttons during update
        self.update_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        
        # Remove the close button from the title bar
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint)
        self.show()  # Need to call show() after changing window flags
    
    def update_progress(self, value):
        """Update the progress bar value"""
        self.progress_bar.setValue(value)
        if value == 100:
            self.status_label.setText("Update downloaded. Preparing to install...")
    
    def closeEvent(self, event):
        """Handle the window close event"""
        if self.update_in_progress:
            event.ignore()
        else:
            event.accept()
    
    def reject(self):
        """Handle dialog rejection (Cancel button or Escape key)"""
        if not self.update_in_progress:
            super().reject()




if __name__ == "__main__":
    # Set up exception handling for the entire application
    def exception_hook(exctype, value, traceback):
        error_msg = f"Uncaught exception: {exctype.__name__}: {value}"
        print(error_msg)
        import traceback as tb
        tb.print_tb(traceback)
        try:
            if DEBUG:
                logger.error(error_msg)
                logger.error("".join(tb.format_tb(traceback)))
        except Exception as e:
            print(f"Error logging exception: {str(e)}")
        sys.__excepthook__(exctype, value, traceback)
        
    # Install the exception hook
    sys.excepthook = exception_hook
    
    # Initialize logging based on debug flag
    try:
        setup_logging(debug_mode=DEBUG)
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
    
    # Log application start
    print(f"Starting Diktando application (Debug mode: {'ON' if DEBUG else 'OFF'})")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if running as frozen executable
    if getattr(sys, 'frozen', False):
        print(f"Running as frozen executable: {sys.executable}")
        print(f"Executable directory: {os.path.dirname(sys.executable)}")
    else:
        print("Running as script")
    
    # Create QApplication first
    try:
        app = QApplication(sys.argv)
        print("QApplication initialized successfully")
    except Exception as e:
        print(f"Error initializing QApplication: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Initialize main window with error handling
    window = None
    try:
        print("Initializing WhisperUI...")
        window = WhisperUI()
        print("WhisperUI initialized successfully")
    except Exception as e:
        error_msg = f"Critical error during initialization: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        try:
            if DEBUG:
                logger.critical(error_msg)
                logger.critical(traceback.format_exc())
        except Exception as log_error:
            print(f"Error logging critical error: {str(log_error)}")
        
        # Show error message to user
        try:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setWindowTitle("Startup Error")
            error_dialog.setText("Failed to initialize application")
            error_dialog.setInformativeText(f"Error: {str(e)}")
            error_dialog.setDetailedText(traceback.format_exc())
            error_dialog.exec_()
        except Exception as dialog_error:
            print(f"Error showing error dialog: {str(dialog_error)}")
        
        sys.exit(1)
    
    # Use a try-finally block to ensure proper cleanup
    try:
        print("Starting application main loop...")
        exit_code = app.exec()
    except Exception as e:
        error_msg = f"Application error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        if DEBUG:
            logger.critical(error_msg)
            logger.critical(traceback.format_exc())
        exit_code = 1
    finally:
        # Ensure all threads are properly stopped before exiting
        print("Application exiting, cleaning up threads...")
        
        try:
            if window:
                # Cancel any active downloads
                if hasattr(window, 'downloader') and window.downloader and window.downloader.isRunning():
                    print("Canceling active download...")
                    window.downloader.cancel()
                    window.downloader.wait(2000)  # Wait up to 2 seconds for it to finish
                
                # Stop any active recording
                if hasattr(window, 'recorder') and window.recorder and window.recorder.isRunning():
                    print("Stopping active recording...")
                    window.recorder.recording = False
                    window.recorder.wait(2000)  # Wait up to 2 seconds for it to finish
                
                # Stop the hotkey manager
                if hasattr(window, 'hotkey_manager') and window.hotkey_manager and window.hotkey_manager.isRunning():
                    print("Stopping hotkey manager...")
                    try:
                        window.hotkey_manager.stop()
                        window.hotkey_manager.wait(2000)  # Wait up to 2 seconds for it to finish
                    except Exception as e:
                        print(f"Error stopping hotkey manager: {str(e)}")
                
                # Stop any active transcription
                if hasattr(window, 'transcriber') and window.transcriber and window.transcriber.isRunning():
                    print("Stopping active transcription...")
                    window.transcriber.wait(2000)  # Wait up to 2 seconds for it to finish
            
            print("Cleanup complete, exiting application")
        except Exception as e:
            error_msg = f"Error during cleanup: {str(e)}"
            print(error_msg)
            if DEBUG:
                logger.error(error_msg)
        
        # Log application end
        if DEBUG:
            logger.info("Application terminated")
    
    sys.exit(exit_code) 