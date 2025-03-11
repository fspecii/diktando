#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import subprocess
import platform
import tempfile
import time
import json
import hashlib
import shutil
from pathlib import Path
import threading
import queue
import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import requests
from tqdm import tqdm
import pyperclip
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QFileDialog, QTextEdit, 
    QProgressBar, QMessageBox, QTabWidget, QGroupBox, QRadioButton,
    QCheckBox, QSpinBox, QSlider, QStatusBar, QKeySequenceEdit, QFrame, QDialog, QButtonGroup,
    QMenu
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QUrl, QEvent
from PyQt5.QtGui import QIcon, QPixmap, QDesktopServices, QKeySequence, QTextCursor

# Import custom modules
from clipboard_manager import ClipboardManager

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
        
    def run(self):
        try:
            print(f"Starting audio recording with device: {self.device}, sample rate: {self.sample_rate}, channels: {self.channels}")
            
            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, 
                               device=self.device, callback=self.audio_callback):
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
            
            # Save to WAV file
            temp_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(temp_dir, "recording.wav")
            
            # Ensure the audio data is in the correct format
            if np.isnan(audio_array).any():
                print("Warning: NaN values detected in audio data, replacing with zeros")
                audio_array = np.nan_to_num(audio_array)
                
            # Normalize audio if it's too quiet
            max_amplitude = np.max(np.abs(audio_array))
            if max_amplitude < 0.1:  # If the maximum amplitude is very low
                print(f"Audio signal is very quiet (max amplitude: {max_amplitude}), normalizing")
                if max_amplitude > 0:  # Avoid division by zero
                    audio_array = audio_array / max_amplitude * 0.5  # Normalize to 50% amplitude
            
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
    
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        # Check if the audio data contains actual sound
        if np.max(np.abs(indata)) < 0.01:
            print("Warning: Very low audio level detected")
            
        self.audio_data.append(indata.copy())
        self.update_signal.emit(indata)
    
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
    hotkey_toggle_signal = pyqtSignal()  # For toggle mode
    hotkey_press_signal = pyqtSignal()   # For push-to-talk press
    hotkey_release_signal = pyqtSignal()  # For push-to-talk release
    error_signal = pyqtSignal(str)
    paste_complete_signal = pyqtSignal(bool)  # True if successful, False otherwise
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hotkey = None
        self.is_push_to_talk = True  # Default to push-to-talk mode
        self.is_pressed = False
        self.transcription = None
        self.clipboard_manager = ClipboardManager()  # Use the enhanced ClipboardManager
        
        # Log the available clipboard formats
        self.log_clipboard_formats()
    
    def log_clipboard_formats(self):
        """Log the available clipboard formats for debugging"""
        try:
            content_types = self.clipboard_manager.get_clipboard_content_type()
            if content_types:
                print(f"Available clipboard formats: {', '.join(content_types)}")
            else:
                print("No clipboard formats available")
        except Exception as e:
            print(f"Error getting clipboard formats: {str(e)}")
    
    def set_mode(self, is_push_to_talk):
        """Set the hotkey mode"""
        self.is_push_to_talk = is_push_to_talk
    
    def set_hotkey(self, hotkey):
        """Set the hotkey to listen for"""
        self.hotkey = hotkey
        
    def run(self):
        """Run the hotkey listener"""
        self.running = True
        
        # Start the fallback timer for key release detection if in push-to-talk mode
        if self.is_push_to_talk:
            self.key_check_timer = threading.Timer(0.1, self._check_key_state)
            self.key_check_timer.daemon = True
            self.key_check_timer.start()
        
        if platform.system() == "Windows":
            self._run_windows_listener()
        else:
            self._run_pynput_listener()
    
    def _check_key_state(self):
        """Fallback mechanism to check if the key is still pressed"""
        try:
            if not self.running:
                return
                
            if self.is_pressed and self.is_push_to_talk and platform.system() == "Windows":
                # Check if the key is still pressed
                hotkey_parts = self.hotkey.lower().split('+')
                
                # For single key hotkeys
                if len(hotkey_parts) == 1:
                    if not keyboard.is_pressed(hotkey_parts[0]):
                        print(f"Fallback detection: {hotkey_parts[0]} is no longer pressed")
                        self.is_pressed = False
                        self._on_key_release(None)
                # For combination hotkeys, check if any part is released
                else:
                    all_pressed = True
                    for part in hotkey_parts:
                        if not keyboard.is_pressed(part):
                            all_pressed = False
                            break
                    
                    if not all_pressed:
                        print(f"Fallback detection: Hotkey combination {self.hotkey} is no longer fully pressed")
                        self.is_pressed = False
                        self._on_key_release(None)
            
            # Schedule the next check
            if self.running and self.is_push_to_talk:
                self.key_check_timer = threading.Timer(0.05, self._check_key_state)  # Check every 50ms instead of 100ms
                self.key_check_timer.daemon = True
                self.key_check_timer.start()
                
        except Exception as e:
            print(f"Error in key state check: {str(e)}")
            # Schedule the next check even if there was an error
            if self.running and self.is_push_to_talk:
                self.key_check_timer = threading.Timer(0.05, self._check_key_state)  # Check every 50ms instead of 100ms
    
    def _run_windows_listener(self):
        """Run the Windows-specific keyboard listener"""
        try:
            print(f"Registering hotkey: {self.hotkey}")
            
            if self.is_push_to_talk:
                # For push-to-talk, register the exact hotkey combination
                keyboard.on_press(lambda e: self._check_hotkey_event(e, True))
                keyboard.on_release(lambda e: self._check_hotkey_event(e, False))
            else:
                # For toggle mode, register the exact hotkey combination
                keyboard.on_press(lambda e: self._check_hotkey_event(e, True))
            
            # Keep the thread running
            while self.running:
                time.sleep(0.1)
            
            # Unregister the hotkey when done
            keyboard.unhook_all()
            
        except Exception as e:
            self.error_signal.emit(f"Hotkey error: {str(e)}")
    
    def _check_hotkey_event(self, event, is_press):
        """Check if the event matches our hotkey combination"""
        try:
            # Get the name of the pressed key
            if hasattr(event, 'name'):
                key_name = event.name
            else:
                key_name = str(event.scan_code)
            
            # Convert to lowercase for case-insensitive comparison
            key_name = key_name.lower()
            hotkey_parts = self.hotkey.lower().split('+')
            
            if is_press:
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
                
                # Construct the current hotkey string
                if modifiers:
                    current_hotkey = '+'.join(modifiers + [key_name])
                else:
                    current_hotkey = key_name
                
                print(f"Press event - Current hotkey: {current_hotkey}, Registered hotkey: {self.hotkey}")
                
                # Check if the current combination matches exactly
                if current_hotkey.lower() == self.hotkey.lower():
                    self.is_pressed = True
                    self._on_key_press(event)
            else:  # Release event
                # For single key hotkeys (like F8)
                if len(hotkey_parts) == 1 and key_name == hotkey_parts[0]:
                    print(f"Release event - Key {key_name} released, matches hotkey {self.hotkey}")
                    if self.is_pressed:
                        self.is_pressed = False
                        self._on_key_release(event)
                # For combination hotkeys (like Ctrl+Space)
                elif key_name in hotkey_parts:
                    print(f"Release event - Key {key_name} released, part of hotkey {self.hotkey}")
                    if self.is_pressed:
                        self.is_pressed = False
                        self._on_key_release(event)
        except Exception as e:
            print(f"Error checking hotkey event: {str(e)}")
            import traceback
            print(traceback.format_exc())
    
    def _run_pynput_listener(self):
        """Run the pynput keyboard listener for macOS and Linux"""
        try:
            # Convert the hotkey string to a key code
            key_code = self._convert_hotkey_to_pynput(self.hotkey)
            
            def on_press(key):
                if hasattr(key, 'char') and key.char == key_code:
                    self.is_pressed = True
                    self._on_key_press(None)
                elif key == key_code:
                    self.is_pressed = True
                    self._on_key_press(None)
            
            def on_release(key):
                if self.is_push_to_talk:
                    if hasattr(key, 'char') and key.char == key_code:
                        if self.is_pressed:
                            self.is_pressed = False
                        self._on_key_release(None)
                    elif key == key_code:
                        if self.is_pressed:
                            self.is_pressed = False
                        self._on_key_release(None)
            
            # Start the listener with both press and release callbacks
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                while self.running:
                    time.sleep(0.1)
                    if not listener.running:
                        break
                        
        except Exception as e:
            self.error_signal.emit(f"Hotkey error: {str(e)}")
    
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
                # Add more as needed
            }
            return special_keys.get(hotkey.lower(), keyboard.Key.f8)  # Default to F8
    
    def _on_key_press(self, event):
        """Handle key press event"""
        if self.is_push_to_talk:
            self.hotkey_press_signal.emit()
        else:
            self.hotkey_toggle_signal.emit()
    
    def _on_key_release(self, event):
        """Handle key release event for push-to-talk mode"""
        if self.is_push_to_talk:
            self.hotkey_release_signal.emit()
    
    def stop(self):
        """Stop the hotkey listener"""
        print("Stopping hotkey listener")
        self.running = False
        
        # Stop the fallback timer
        if self.key_check_timer:
            self.key_check_timer.cancel()
        
        # Unregister hotkeys on Windows
        if platform.system() == "Windows":
            try:
                keyboard.unhook_all()
                print("Unhooked all keyboard listeners")
            except Exception as e:
                print(f"Error unhooking keyboard listeners: {str(e)}")
        
        # For other platforms, the listener will stop when the thread ends
        # because we're using a context manager with the keyboard.Listener
    
    def backup_clipboard(self):
        """Backup the current clipboard content"""
        self.clipboard_manager.backup_clipboard()
    
    def restore_clipboard(self):
        """Restore the original clipboard content"""
        self.clipboard_manager.restore_clipboard()
    
    def set_transcription(self, text):
        """Set the transcription text"""
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

    def force_release(self):
        """Force a key release event (for debugging or recovery)"""
        print("Forcing hotkey release event")
        if self.is_pressed:
            self.is_pressed = False
            self._on_key_release(None)
            return True
        else:
            print("Hotkey not currently pressed, nothing to release")
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
        """Initialize the WhisperUI application"""
        super().__init__()
        
        # Set application icon
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            application_path = sys._MEIPASS
        else:
            # Running in normal Python environment
            application_path = os.path.dirname(os.path.abspath(__file__))
            
        icon_path = os.path.join(application_path, "new_icon.ico")
        print(f"Loading icon from: {icon_path}")
        if os.path.exists(icon_path):
            print("Icon file exists, setting window icon...")
            icon = QIcon(icon_path)
            if not icon.isNull():
                print("Icon loaded successfully")
                self.setWindowIcon(icon)
                # Also set the taskbar icon
                if platform.system() == "Windows":
                    print("Setting Windows taskbar icon...")
                    import ctypes
                    myappid = 'diktando.speech.recognition.1.0'
                    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            else:
                print("Failed to load icon - icon is null")
        else:
            print(f"Icon file not found at {icon_path}")
        
        # Set up the UI
        self.setWindowTitle("Diktando")
        self.setMinimumSize(800, 600)
        
        # Create status bar first
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Initializing...")
        
        # Create overlay window first
        self.create_overlay_window()
        
        # Initialize state flags
        self.is_hotkey_recording = False
        self.is_recording = False
        self.is_transcribing = False
        self.is_downloading = False
        
        # Initialize transcription history
        self.transcription_history = []
        
        # Initialize timers
        self.hotkey_recording_timer = QTimer(self)
        self.hotkey_recording_timer.timeout.connect(self.update_hotkey_recording_indicator)
        self.hotkey_recording_timer.setInterval(500)  # Update every 500ms
        
        # Check and download required binary files
        if not self.check_required_binaries():
            self.show_error("Failed to download required files. Please check your internet connection and try again.")
            return
        
        # Initialize clipboard manager
        self.clipboard_manager = ClipboardManager()
        
        # Initialize hotkey manager
        self.hotkey_manager = HotkeyManager(self)
        self.hotkey_manager.hotkey_toggle_signal.connect(self.on_hotkey_toggle)
        self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
        self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
        self.hotkey_manager.error_signal.connect(self.show_error)
        self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)
        
        # Initialize the UI components
        self.init_ui()
        self.init_models()
        self.init_audio_devices()
        
        # Initialize thread objects
        self.recorder = None
        self.transcriber = None
        self.downloader = None
        self.current_audio_file = None
        
        # Enable dark mode by default
        self.dark_mode = True
        self.toggle_dark_mode(True)
        if hasattr(self, 'dark_mode_check'):
            self.dark_mode_check.setChecked(True)
        
        # Load saved settings or use defaults
        settings_loaded = self.load_settings()
        if not settings_loaded:
            # Set push-to-talk as default if no settings were loaded
            if hasattr(self, 'push_to_talk_radio'):
                self.push_to_talk_radio.setChecked(True)
                self.toggle_mode_radio.setChecked(False)
                self.on_mode_changed()
        
        # Check if any models are installed and download default if needed
        self.check_and_download_default_model()
        
        # Show the main window
        self.show()
        
        # Enable hotkey by default with a slight delay to ensure UI is fully loaded
        QTimer.singleShot(500, self.enable_hotkey_by_default)
        
        # Initialize recording paths
        self.recording_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recording.wav")
        self.debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_recordings")
        self.silence_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "silence.wav")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Generate silence file if it doesn't exist
        if not os.path.exists(self.silence_path):
            self.generate_silence_file()
    
    def __del__(self):
        """Destructor to clean up resources"""
        print("WhisperUI destructor called, cleaning up resources...")
        self.cleanup_resources()
    
    def cleanup_resources(self):
        """Clean up all resources and threads"""
        print("Cleaning up resources...")
        
        # Stop any active timers
        if hasattr(self, 'hotkey_recording_timer') and self.hotkey_recording_timer.isActive():
            self.hotkey_recording_timer.stop()
        
        if hasattr(self, 'overlay_animation_timer') and self.overlay_animation_timer.isActive():
            self.overlay_animation_timer.stop()
        
        if hasattr(self, 'fade_timer') and self.fade_timer.isActive():
            self.fade_timer.stop()
        
        if hasattr(self, 'fade_out_timer') and self.fade_out_timer.isActive():
            self.fade_out_timer.stop()
        
        # Cancel any active downloads
        if hasattr(self, 'downloader') and self.downloader and self.downloader.isRunning():
            print("Canceling active download...")
            self.downloader.cancel()
            self.downloader.wait(2000)  # Wait up to 2 seconds for it to finish
        
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
        
        print("Resource cleanup complete")
    
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
        self.enable_hotkey_check.setChecked(True)
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Create transcription tab
        transcription_tab = QWidget()
        tabs.addTab(transcription_tab, "Transcription")
        
        # Create model management tab
        model_tab = QWidget()
        tabs.addTab(model_tab, "Models")
        
        # Create settings tab
        settings_tab = QWidget()
        tabs.addTab(settings_tab, "Settings")
        
        # Set up transcription tab
        self.setup_transcription_tab(transcription_tab)
        
        # Set up model management tab
        self.setup_model_tab(model_tab)
        
        # Set up settings tab
        self.setup_settings_tab(settings_tab)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
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
        """Set up the settings tab"""
        
        layout = QVBoxLayout(tab)
        
        # Audio settings
        audio_group = QGroupBox("Audio Settings")
        audio_layout = QVBoxLayout(audio_group)
        
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(QLabel("Sample Rate:"))
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems(["16000", "22050", "44100", "48000"])
        self.sample_rate_combo.setCurrentText("16000")
        sample_rate_layout.addWidget(self.sample_rate_combo)
        audio_layout.addLayout(sample_rate_layout)
        
        channels_layout = QHBoxLayout()
        channels_layout.addWidget(QLabel("Channels:"))
        self.channels_combo = QComboBox()
        self.channels_combo.addItems(["1 (Mono)", "2 (Stereo)"])
        self.channels_combo.setCurrentIndex(0)
        channels_layout.addWidget(self.channels_combo)
        audio_layout.addLayout(channels_layout)
        
        layout.addWidget(audio_group)
        
        # Hotkey settings
        hotkey_group = QGroupBox("Hotkey Settings")
        hotkey_layout = QVBoxLayout(hotkey_group)
        
        self.enable_hotkey_check = QCheckBox("Enable Global Hotkey")
        self.enable_hotkey_check.toggled.connect(self.toggle_hotkey)
        hotkey_layout.addWidget(self.enable_hotkey_check)
        
        # Recording mode group
        mode_group = QGroupBox("Recording Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(5)
        mode_layout.setContentsMargins(8, 8, 8, 8)
        
        # Create button group for mutual exclusivity
        button_group = QButtonGroup(mode_group)
        
        # Toggle mode option with more compact layout
        toggle_widget = QWidget()
        toggle_layout = QHBoxLayout(toggle_widget)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        
        self.toggle_mode_radio = QRadioButton("Toggle Mode")
        self.toggle_mode_radio.setChecked(True)
        self.toggle_mode_radio.toggled.connect(self.on_mode_changed)
        button_group.addButton(self.toggle_mode_radio)
        toggle_layout.addWidget(self.toggle_mode_radio)
        
        toggle_description = QLabel("(click to start/stop)")
        toggle_description.setStyleSheet("color: gray;")
        toggle_layout.addWidget(toggle_description)
        toggle_layout.addStretch()
        
        mode_layout.addWidget(toggle_widget)
        
        # Push-to-talk option with more compact layout
        ptt_widget = QWidget()
        ptt_layout = QHBoxLayout(ptt_widget)
        ptt_layout.setContentsMargins(0, 0, 0, 0)
        
        self.push_to_talk_radio = QRadioButton("Push-to-Talk Mode")
        self.push_to_talk_radio.toggled.connect(self.on_mode_changed)
        button_group.addButton(self.push_to_talk_radio)
        ptt_layout.addWidget(self.push_to_talk_radio)
        
        ptt_description = QLabel("(hold to record)")
        ptt_description.setStyleSheet("color: gray;")
        ptt_layout.addWidget(ptt_description)
        ptt_layout.addStretch()
        
        mode_layout.addWidget(ptt_widget)
        
        hotkey_layout.addWidget(mode_group)
        
        # Hotkey configuration
        hotkey_config_layout = QHBoxLayout()
        hotkey_config_layout.addWidget(QLabel("Current Hotkey:"))
        
        self.hotkey_label = QLabel("F8")  # Default hotkey
        self.hotkey_label.setStyleSheet("font-weight: bold;")
        hotkey_config_layout.addWidget(self.hotkey_label)
        
        configure_button = QPushButton("Configure")
        configure_button.clicked.connect(self.show_hotkey_config)
        hotkey_config_layout.addWidget(configure_button)
        hotkey_config_layout.addStretch()
        
        hotkey_layout.addLayout(hotkey_config_layout)
        
        # Add auto-paste option
        self.auto_paste_check = QCheckBox("Automatically paste transcription (may require permissions)")
        self.auto_paste_check.setChecked(True)
        self.auto_paste_check.setToolTip("When disabled, transcription will only be copied to clipboard without pasting")
        hotkey_layout.addWidget(self.auto_paste_check)
        
        # Add hotkey info
        hotkey_info = QLabel(
            "Press the hotkey once to start recording, press again to stop and transcribe.\n"
            "The transcription will automatically replace your clipboard content\n"
            "and be pasted at the cursor position (if auto-paste is enabled).\n\n"
            "When recording is active, the window title will show 'RECORDING'.\n"
            "If you experience transcription errors, try speaking more clearly and\n"
            "ensure your microphone is working properly."
        )
        hotkey_info.setWordWrap(True)
        hotkey_layout.addWidget(hotkey_info)
        
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
        
        # About section
        about_group = QGroupBox("About")
        about_layout = QVBoxLayout(about_group)
        
        about_text = QLabel(
        "Diktando\n"
            "A Windows GUI for OpenAI's Whisper speech recognition model\n"
            "Using whisper-cpp for local transcription\n"
            "Version: 1.0.0\n"
        )
        about_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        about_layout.addWidget(about_text)
        
        github_button = QPushButton("Visit GitHub Repository")
        github_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl("https://github.com/fspecii/diktando")
        ))
        about_layout.addWidget(github_button)
        
        layout.addWidget(about_group)
    
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
            sample_rate = int(self.sample_rate_combo.currentText())
            channels = 1 if self.channels_combo.currentIndex() == 0 else 2
            
            self.recorder = AudioRecorder(sample_rate, channels, device_idx)
            self.recorder.update_signal.connect(self.update_level_meter)
            self.recorder.finished_signal.connect(self.recording_finished)
            self.recorder.error_signal.connect(self.show_error)
            
            self.recorder.start_recording()
            
            self.record_button.setText("Stop")
            self.status_bar.showMessage("Recording...")
            
        except Exception as e:
            self.show_error(f"Failed to start recording: {str(e)}")
    
    def stop_recording(self):
        """Stop recording audio"""
        if self.recorder and self.recorder.recording:
            self.recorder.stop_recording()
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
        if enabled:
            # Get the current hotkey
            hotkey = self.hotkey_label.text()
            if not hotkey:
                self.show_error("Please configure a hotkey first")
                self.enable_hotkey_check.setChecked(False)
                return
            
            # Convert QKeySequence to a string format that keyboard library can understand
            hotkey_str = self._convert_key_sequence_to_string(QKeySequence(hotkey))
            
            # Stop existing hotkey manager if running
            if self.hotkey_manager.isRunning():
                print("Stopping existing hotkey manager")
                self.hotkey_manager.stop()
                self.hotkey_manager.wait()  # Wait for thread to finish
                
                # Create a new hotkey manager since the old one can't be reused
                self.hotkey_manager = HotkeyManager(self)
                self.hotkey_manager.hotkey_toggle_signal.connect(self.on_hotkey_toggle)
                self.hotkey_manager.hotkey_press_signal.connect(self.on_hotkey_press)
                self.hotkey_manager.hotkey_release_signal.connect(self.on_hotkey_release)
                self.hotkey_manager.error_signal.connect(self.show_error)
                self.hotkey_manager.paste_complete_signal.connect(self.on_paste_complete)
            
            # Set the mode and hotkey
            self.hotkey_manager.set_mode(self.push_to_talk_radio.isChecked())
            self.hotkey_manager.set_hotkey(hotkey_str)
            self.hotkey_manager.start()
            
            mode_str = "Push-to-Talk" if self.push_to_talk_radio.isChecked() else "Toggle"
            self.status_bar.showMessage(f"Hotkey enabled: {hotkey} ({mode_str} mode)")
        else:
            # Stop the hotkey manager
            if self.hotkey_manager.isRunning():
                print("Stopping hotkey manager")
                self.hotkey_manager.stop()
                self.hotkey_manager.wait()  # Wait for thread to finish
            
            self.status_bar.showMessage("Hotkey disabled")

    def on_hotkey_press(self):
        """Handle hotkey press in push-to-talk mode"""
        print("HOTKEY PRESS DETECTED!")
        if not self.is_hotkey_recording:
            print("Starting recording via hotkey press")
            self.start_hotkey_recording()
        else:
            print("Hotkey pressed but already recording - ignoring")

    def on_hotkey_release(self):
        """Handle hotkey release in push-to-talk mode"""
        print("HOTKEY RELEASE DETECTED!")
        if self.is_hotkey_recording:
            print("Hotkey released - stopping recording and starting transcription")
            # Stop recording - this will trigger the finished_signal which calls hotkey_recording_finished
            self.stop_hotkey_recording()
            # Update status
            self.status_bar.showMessage("Processing recording...")
        else:
            print("Hotkey released but not recording - ignoring")

    def show_hotkey_config(self):
        """Show the hotkey configuration dialog"""
        current_hotkey = self.hotkey_label.text()
        dialog = HotkeyConfigDialog(self, current_hotkey)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_hotkey = dialog.get_hotkey()
            if new_hotkey:
                # Update the label
                self.hotkey_label.setText(new_hotkey)
                
                # Save the settings
                self.save_settings()
                
                # If hotkey is enabled, restart with new hotkey
                if self.enable_hotkey_check.isChecked():
                    self.toggle_hotkey(False)  # Stop current hotkey
                    self.toggle_hotkey(True)   # Start with new hotkey
                
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
    
    def on_hotkey_toggle(self):
        """Handle hotkey toggle event - starts or stops recording"""
        try:
            if not self.is_hotkey_recording:
                # Start recording
                self.start_hotkey_recording()
            else:
                # Stop recording
                self.stop_hotkey_recording()
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
        # Add to history and get cleaned text
        cleaned_text = self.append_to_transcription_history(text, source="hotkey")
        
        # Set the transcription and attempt to paste it
        self.hotkey_manager.set_transcription(cleaned_text)
        self.show_overlay("Pasting transcription...")
        self.attempt_paste_transcription()
        
        # Log the transcription
        self.log_message(f"Transcription copied to clipboard: {cleaned_text}")

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
        """Restore the original clipboard content"""
        try:
            self.hotkey_manager.restore_clipboard()
            
            # Get the content types after restoration
            content_types = self.clipboard_manager.get_clipboard_content_type()
            content_type_str = ", ".join(content_types) if content_types else "empty"
            print(f"Restored clipboard content types: {content_type_str}")
            
            self.status_bar.showMessage("Original clipboard content restored")
        except Exception as e:
            self.status_bar.showMessage(f"Error restoring clipboard: {str(e)}")
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
        """Clean up the transcription text"""
        if not text:
            return text
            
        # Remove [Silence] markers
        text = re.sub(r'\[Silence\]\s*', '', text)
        
        if not keep_timestamps:
            # Remove timestamps if present
            text = re.sub(r'\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]\s*', '', text)
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text

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
        """Load settings from file"""
        settings_file = self.get_settings_file()
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    # Load hotkey
                    if 'hotkey' in settings:
                        self.hotkey_label.setText(settings['hotkey'])
                    # Load mode
                    if 'push_to_talk' in settings:
                        self.push_to_talk_radio.setChecked(settings['push_to_talk'])
                        self.toggle_mode_radio.setChecked(not settings['push_to_talk'])
                    # Load dark mode setting
                    if 'dark_mode' in settings:
                        self.dark_mode_check.setChecked(settings['dark_mode'])
                        # Apply dark mode if enabled
                        if settings['dark_mode']:
                            self.toggle_dark_mode(True)
            except Exception as e:
                print(f"Error loading settings: {str(e)}")
    
    def save_settings(self):
        """Save settings to file"""
        settings_file = self.get_settings_file()
        try:
            settings = {
                'hotkey': self.hotkey_label.text(),
                'push_to_talk': self.push_to_talk_radio.isChecked(),
                'dark_mode': self.dark_mode_check.isChecked()
            }
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
        
        # Restart the hotkey manager with new mode
        self.toggle_hotkey(False)  # Stop current
        self.toggle_hotkey(True)   # Start with new mode
        
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
            if "press the hotkey" in child.text().lower():
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

    def manual_trigger_release(self):
        """Manually trigger the hotkey release event for debugging"""
        print("Manually triggering hotkey release event")
        if self.is_hotkey_recording:
            print("Recording is active, manually triggering release event")
            
            # Try to force release via the hotkey manager first
            if self.hotkey_manager and self.hotkey_manager.isRunning():
                if self.hotkey_manager.force_release():
                    print("Force release successful via hotkey manager")
                    return
            
            # If that didn't work, call the release handler directly
            print("Calling on_hotkey_release directly")
            self.on_hotkey_release()
        else:
            print("Not currently recording, nothing to release")
            self.status_bar.showMessage("Not recording - nothing to release")

    def log_message(self, message):
        """Add a message to the debug log"""
        timestamp = time.strftime("%H:%M:%S")
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
        
        # Don't clean the text immediately to preserve speech timestamps
        history_entry = f"[{timestamp}] - {source.upper()}\n{text}\n{'-' * 80}\n\n"
        
        # Add to history list
        self.transcription_history.append(history_entry)
        
        # Update the results text area with all history
        self.results_text.setPlainText("".join(self.transcription_history))
        
        # Scroll to the bottom to show the latest transcription
        self.results_text.moveCursor(QTextCursor.End)
        
        # Return cleaned version only for clipboard operations
        return self.clean_transcription(text) if source == "hotkey" else text

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




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhisperUI()
    
    # Use a try-finally block to ensure proper cleanup
    try:
        exit_code = app.exec()
    finally:
        # Ensure all threads are properly stopped before exiting
        print("Application exiting, cleaning up threads...")
        
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
            window.hotkey_manager.stop()
            window.hotkey_manager.wait(2000)  # Wait up to 2 seconds for it to finish
        
        # Stop any active transcription
        if hasattr(window, 'transcriber') and window.transcriber and window.transcriber.isRunning():
            print("Stopping active transcription...")
            window.transcriber.wait(2000)  # Wait up to 2 seconds for it to finish
        
        print("Cleanup complete, exiting application")
    
    sys.exit(exit_code) 