import os
import json
import base64
import io
try:
    import google.generativeai as genai
except ImportError:
    # Try alternative import path
    import sys
    import site
    # Add user-specific site-packages to path
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)
    try:
        import google.generativeai as genai
    except ImportError as e:
        print(f"Error importing google.generativeai: {e}")
        print("Python path:", sys.path)
        raise

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap

class LLMProcessor(QObject):
    """Handles LLM processing with different providers"""
    processing_complete = pyqtSignal(str)  # Processed text
    processing_error = pyqtSignal(str)  # Error message
    processing_started = pyqtSignal()  # Signal when processing starts
    processing_stopped = pyqtSignal()  # Signal when processing stops

    def __init__(self):
        super().__init__()
        self.provider = "gemini"  # Default provider
        self.api_key = ""
        self.model = "gemini-1.5-pro"  # Updated to a model that supports vision
        self.prompt_template = "Process and improve the following transcription: {text}"
        self.is_push_to_talk = True  # Default to push-to-talk mode
        self.is_processing = False
        self.include_screenshot = False  # Default to not include screenshots
        self.screenshot_data = None  # Will store the screenshot data when captured
        self.load_settings()

    def load_settings(self):
        """Load LLM settings from file"""
        settings_file = self.get_settings_file()
        if os.path.exists(settings_file):
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.provider = settings.get('provider', 'gemini')
                    self.api_key = settings.get('api_key', '')
                    self.model = settings.get('model', 'gemini-1.5-pro')
                    self.prompt_template = settings.get('prompt_template', self.prompt_template)
                    self.is_push_to_talk = settings.get('is_push_to_talk', True)
                    self.include_screenshot = settings.get('include_screenshot', False)
            except Exception as e:
                print(f"Error loading LLM settings: {str(e)}")

    def save_settings(self):
        """Save LLM settings to file"""
        settings_file = self.get_settings_file()
        try:
            settings = {
                'provider': self.provider,
                'api_key': self.api_key,
                'model': self.model,
                'prompt_template': self.prompt_template,
                'is_push_to_talk': self.is_push_to_talk,
                'include_screenshot': self.include_screenshot
            }
            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            with open(settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Error saving LLM settings: {str(e)}")

    def get_settings_file(self):
        """Get the path to the settings file"""
        app_data = os.path.join(os.path.expanduser('~'), '.diktando')
        return os.path.join(app_data, 'llm_settings.json')

    def set_provider(self, provider, api_key, model=None):
        """Set the LLM provider and API key"""
        self.provider = provider
        self.api_key = api_key
        if model:
            self.model = model
        self.save_settings()

    def set_prompt_template(self, template):
        """Set the prompt template"""
        self.prompt_template = template
        self.save_settings()

    def set_mode(self, is_push_to_talk):
        """Set the processing mode"""
        self.is_push_to_talk = is_push_to_talk
        self.save_settings()

    def set_include_screenshot(self, include_screenshot):
        """Set whether to include screenshots in LLM processing"""
        self.include_screenshot = include_screenshot
        self.save_settings()

    def capture_screenshot(self):
        """Capture a screenshot of the primary screen"""
        try:
            print("Attempting to capture screenshot...")
            
            # Get the primary screen
            screen = QApplication.primaryScreen()
            if not screen:
                print("Error: Could not get primary screen")
                return False
                
            print("Got primary screen, capturing window...")
            
            # Capture the entire screen
            pixmap = screen.grabWindow(0)  # 0 means the entire screen
            
            if pixmap.isNull():
                print("Error: Failed to capture screenshot")
                return False
                
            print(f"Screenshot captured successfully. Size: {pixmap.width()}x{pixmap.height()}")
            
            # Create a temporary file to save the screenshot
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            
            print(f"Created temporary file: {temp_file_path}")
            
            # Save the pixmap to the temporary file
            if not pixmap.save(temp_file_path, "PNG"):
                print(f"Error: Failed to save screenshot to {temp_file_path}")
                return False
                
            print(f"Screenshot saved to temporary file")
            
            # Read the file as bytes
            with open(temp_file_path, 'rb') as f:
                image_data = f.read()
                
            # Encode to base64
            self.screenshot_data = base64.b64encode(image_data).decode('utf-8')
            print(f"Screenshot encoded to base64. Length: {len(self.screenshot_data)} characters")
            
            # Clean up the temporary file
            try:
                import os
                os.unlink(temp_file_path)
                print(f"Temporary file deleted")
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_file_path}: {str(e)}")
                
            return True
        except Exception as e:
            print(f"Error capturing screenshot: {str(e)}")
            import traceback
            traceback.print_exc()
            self.screenshot_data = None
            return False

    def start_processing(self, text):
        """Start processing text with LLM"""
        if not self.api_key:
            self.processing_error.emit("No API key configured. Please set up LLM settings first.")
            return
            
        if not text:
            self.processing_error.emit("No text to process")
            return
            
        self.is_processing = True
        self.processing_started.emit()
        
        try:
            # Configure Gemini
            print(f"Configuring Gemini with API key (length: {len(self.api_key)})")
            genai.configure(api_key=self.api_key)
            
            # Get the model
            print(f"Using Gemini model: {self.model}")
            model = genai.GenerativeModel(self.model)
            
            # Format the prompt
            prompt = self.prompt_template.format(text=text)
            print(f"Formatted prompt (first 100 chars): {prompt[:100]}...")
            
            # Capture screenshot if enabled
            screenshot_included = False
            if self.include_screenshot:
                print("Screenshot inclusion is enabled, attempting to capture...")
                if self.capture_screenshot():
                    screenshot_included = True
                    print("Screenshot captured successfully")
                else:
                    print("Failed to capture screenshot")
            else:
                print("Screenshot inclusion is disabled")
                
            # Generate response based on whether screenshot is included
            if screenshot_included and self.screenshot_data:
                # Create multipart content with text and image
                print("Creating multipart content with text and image")
                image_parts = [
                    {
                        "text": prompt
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": self.screenshot_data
                        }
                    }
                ]
                print(f"Sending request with text and image to Gemini")
                response = model.generate_content(image_parts)
            else:
                # Text-only response
                print(f"Sending text-only request to Gemini")
                response = model.generate_content(prompt)
            
            print(f"Received response from Gemini")
            if response.text:
                print(f"Response text (first 100 chars): {response.text[:100]}...")
                self.processing_complete.emit(response.text)
            else:
                print("Gemini returned empty response")
                self.processing_error.emit("LLM returned empty response")
        except Exception as e:
            print(f"LLM processing error: {str(e)}")
            import traceback
            traceback.print_exc()
            self.processing_error.emit(f"LLM processing error: {str(e)}")
        finally:
            self.is_processing = False
            self.processing_stopped.emit()
            # Clear screenshot data after processing
            self.screenshot_data = None
            print("LLM processing completed, screenshot data cleared")

    def stop_processing(self):
        """Stop current processing"""
        if self.is_processing:
            self.is_processing = False
            self.processing_stopped.emit()

    async def process_text(self, text):
        """Process text with the configured LLM"""
        if not self.api_key:
            raise ValueError("No API key configured")
            
        if not text:
            raise ValueError("No text to process")
            
        if self.provider == "gemini":
            return await self._process_with_gemini(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _process_with_gemini(self, text):
        """Process text using Gemini"""
        try:
            # Configure Gemini
            genai.configure(api_key=self.api_key)
            
            # Get the model
            model = genai.GenerativeModel(self.model)
            
            # Format the prompt
            prompt = self.prompt_template.format(text=text)
            
            # Capture screenshot if enabled
            screenshot_included = False
            if self.include_screenshot and self.capture_screenshot():
                screenshot_included = True
                
            # Generate response based on whether screenshot is included
            if screenshot_included and self.screenshot_data:
                # Create multipart content with text and image
                image_parts = [
                    {
                        "text": prompt
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": self.screenshot_data
                        }
                    }
                ]
                response = model.generate_content(image_parts)
            else:
                # Text-only response
                response = model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                raise ValueError("Gemini returned empty response")
        except Exception as e:
            raise Exception(f"Gemini processing error: {str(e)}")

    def get_available_models(self):
        """Get list of available models for current provider"""
        if self.provider == "gemini":
            return ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro-vision"]
        return [] 