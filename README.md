# Diktando - Advanced Speech Recognition & AI Processing Tool

![Diktando Logo](icon.ico)

Diktando is a professional-grade desktop application that combines state-of-the-art speech recognition with AI-powered text processing. Designed for productivity and ease of use, Diktando transforms your spoken words into polished text with just a keystroke.

## Key Features

### Speech Recognition
- **High-Accuracy Transcription**: Powered by OpenAI's Whisper models for industry-leading speech recognition
- **Real-time Processing**: Convert speech to text instantly with minimal delay
- **Multi-language Support**: Transcribe content in multiple languages with excellent accuracy
- **Customizable Models**: Choose from various model sizes (tiny to large) based on your accuracy needs

### AI Text Processing
- **LLM Integration**: Process transcribed text through Google's Gemini AI models
- **Customizable Prompts**: Define how the AI should process your text (formatting, grammar correction, summarization)
- **Seamless Workflow**: Capture speech with F8, process with AI using F9, and paste the result automatically

### User Experience
- **Global Hotkey Support**: Control recording from any application with customizable hotkeys
- **Dual Recording Modes**:
  - Toggle Mode: Press once to start, again to stop
  - Push-to-Talk: Hold to record, release to stop
- **Visual Feedback**: Overlay indicators show recording status without disrupting workflow
- **Dark/Light Themes**: Choose your preferred visual style

### Clipboard Management
- **Intelligent Clipboard Handling**: Automatically manages clipboard content during operations
- **Format Preservation**: Maintains formatting when possible
- **Backup & Restore**: Preserves your original clipboard content

### Export & Documentation
- **Multiple Export Formats**:
  - Plain Text (.txt) for maximum compatibility
  - PDF documents with formatting
  - SRT subtitles with timestamps
- **Transcription History**: Review and access previous transcriptions
- **Session Persistence**: History is preserved between application sessions

## Getting Started

### Installation
1. Download the latest release from the [releases page](https://github.com/fspecii/diktando/releases)
2. Run the executable `DiktandoApp_v1.2.exe`
3. The application will automatically download the base English model on first run

### System Requirements
- Windows 10 or later
- 4GB RAM minimum (8GB recommended for larger models)
- 500MB free disk space for the base model
- Microphone for audio recording
- Internet connection for LLM processing and model downloads

### Quick Start Guide
1. **Basic Transcription**:
   - Press F8 (default hotkey) to start/stop recording
   - Speak clearly into your microphone
   - The recorded audio will be automatically transcribed
   - The transcription appears in the application and is copied to clipboard

2. **AI Processing**:
   - Press F9 to record speech for AI processing
   - The text is transcribed and then sent to the configured LLM
   - Processed text is automatically copied to clipboard
   - Configure LLM settings in the Settings tab

3. **Configuration**:
   - **Hotkeys**: Customize recording and LLM hotkeys in Settings
   - **Models**: Download additional Whisper models from the Models tab
   - **LLM Settings**: Configure your Gemini API key and prompt templates
   - **Audio**: Select input device and adjust recording parameters

## Advanced Usage

### LLM Configuration
1. Obtain a Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Enter your API key in the LLM Settings dialog
3. Customize the prompt template to control how your text is processed
4. Choose between Gemini 1.0 Pro or Gemini 2.0 Flash models

### Model Selection Guide
| Model | Size | Speed | Accuracy | Recommended Use |
|-------|------|-------|----------|-----------------|
| Tiny  | 75MB | Fast  | Basic    | Quick notes, short commands |
| Base  | 142MB| Good  | Good     | General transcription |
| Small | 466MB| Medium| Very Good| Professional documents |
| Medium| 1.5GB| Slow  | Excellent| Critical accuracy needs |
| Large | 3GB  | Slowest| Superior| Research, legal, medical |

### Keyboard Shortcuts
- **F8**: Start/stop basic transcription
- **F9**: Start/stop LLM-enhanced transcription
- **Ctrl+C**: Copy selected text
- **Ctrl+S**: Save transcription
- **Ctrl+D**: Toggle dark/light mode

## Development

### Prerequisites
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Building from Source
```bash
# Build executable
pyinstaller diktando.spec
```

### Project Structure
- `main.py` - Main application code
- `clipboard_manager.py` - Clipboard operations
- `llm_processor.py` - AI text processing functionality
- `llm_settings_dialog.py` - UI for LLM configuration
- `updater.py` - Application update functionality
- `diktando.spec` - PyInstaller specification
- `requirements.txt` - Python dependencies

## License

This project is licensed under the Diktando Non-Commercial License - see the [LICENSE](LICENSE) file for details.

**Important**: This software is free for personal, educational, and non-commercial use only. Commercial use requires explicit permission from the copyright holder.

## Acknowledgments

- OpenAI's Whisper for the speech recognition models
- Google's Gemini for AI text processing
- PyQt5 for the GUI framework
- The open-source community for various libraries and tools

---

© 2024 Diktando | [GitHub Repository](https://github.com/fspecii/diktando) 