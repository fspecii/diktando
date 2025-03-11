# Diktando - Speech Recognition Tool

Diktando is a powerful desktop application that provides real-time speech recognition and transcription capabilities using OpenAI's Whisper models. With a clean, modern interface and global hotkey support, it makes transcription tasks effortless.

## Features

- **Real-time Speech Recognition**: Convert speech to text in real-time using state-of-the-art Whisper models
- **Global Hotkey Support**: Start/stop recording from anywhere with customizable hotkeys
- **Multiple Recording Modes**:
  - Toggle Mode: Click to start/stop recording
  - Push-to-Talk Mode: Hold key to record, release to stop
- **Clipboard Integration**:
  - Automatic clipboard backup and restoration
  - Smart paste functionality
  - Support for multiple data types (text, images, HTML)
- **Multiple Export Formats**:
  - Plain Text (.txt)
  - PDF documents
  - SRT subtitles
- **Customizable Settings**:
  - Multiple language support
  - Adjustable audio settings (sample rate, channels)
  - Dark/Light theme
  - Configurable model directory

## Installation

1. Download the latest release from the releases page
2. Run the installer or extract the portable version
3. Launch `DiktandoApp_v1.exe`

The application will automatically download the base English model (approximately 140MB) on first run.

## System Requirements

- Windows 10 or later
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space for the base model
- Microphone for audio recording

## Usage

1. **Basic Recording**:
   - Press F8 (default hotkey) to start/stop recording
   - The recorded audio will be automatically transcribed

2. **Configuration**:
   - Go to the Settings tab to customize hotkeys
   - Choose your preferred recording mode
   - Select audio input device and settings

3. **Models**:
   - Download additional models from the Models tab
   - Available models: tiny, base, small, medium, large
   - Each model offers different accuracy/speed trade-offs

4. **Export**:
   - Use the export buttons to save transcriptions
   - Choose from TXT, PDF, or SRT formats
   - History is preserved between sessions

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
- `diktando.spec` - PyInstaller specification
- `requirements.txt` - Python dependencies

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI's Whisper for the speech recognition models
- PyQt5 for the GUI framework
- The open-source community for various libraries and tools 