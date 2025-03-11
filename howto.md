# Building Diktando

To build the Diktando application, follow these steps:

1. Make sure all dependencies are installed:
   ```
   pip install -r requirements.txt
   ```

2. Build the application using PyInstaller:
   ```
   PyInstaller --clean diktando.spec
   ```

3. The built application will be available in the `dist` directory.

## Project Structure

- `main.py` - Main application code
- `clipboard_manager.py` - Module for clipboard operations
- `diktando.spec` - PyInstaller specification file
- `requirements.txt` - Python dependencies
- `icon.ico` - Application icon

## Enhanced Clipboard Manager

The clipboard manager now supports multiple data types:

- Text content
- Images
- HTML content
- URLs

When using the hotkey functionality, the clipboard manager will:
1. Backup the current clipboard content (including images)
2. Set the transcription text to the clipboard
3. Paste the transcription if auto-paste is enabled
4. Restore the original clipboard content (including images)

This ensures that users don't lose their previous clipboard content, even if it contained non-text data like images.