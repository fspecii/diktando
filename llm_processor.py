import os
import json
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
        self.model = "gemini-2.0-flash"  # Gemini 2.0 Flash model
        self.prompt_template = "Process and improve the following transcription: {text}"
        self.is_push_to_talk = True  # Default to push-to-talk mode
        self.is_processing = False
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
                    self.model = settings.get('model', 'gemini-2.0-flash')
                    self.prompt_template = settings.get('prompt_template', self.prompt_template)
                    self.is_push_to_talk = settings.get('is_push_to_talk', True)
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
                'is_push_to_talk': self.is_push_to_talk
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
            genai.configure(api_key=self.api_key)
            
            # Get the model
            model = genai.GenerativeModel(self.model)
            
            # Format the prompt
            prompt = self.prompt_template.format(text=text)
            
            # Generate response
            response = model.generate_content(prompt)
            
            if response.text:
                self.processing_complete.emit(response.text)
            else:
                self.processing_error.emit("LLM returned empty response")
        except Exception as e:
            self.processing_error.emit(f"LLM processing error: {str(e)}")
        finally:
            self.is_processing = False
            self.processing_stopped.emit()

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
            
            # Generate response
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
            return ["gemini-2.0-flash", "gemini-1.0-pro"]
        return [] 