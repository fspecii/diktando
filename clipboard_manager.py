#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import platform
import pyperclip
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QMimeData
import keyboard

class ClipboardManager:
    """Manages clipboard operations for the application"""
    
    def __init__(self):
        """Initialize the clipboard manager"""
        self.original_mime_data = None
        self.clipboard = QApplication.clipboard()
    
    def backup_clipboard(self):
        """Backup the current clipboard content including text and images"""
        try:
            mime_data = self.clipboard.mimeData()
            if mime_data:
                self.original_mime_data = QMimeData()
                for format in mime_data.formats():
                    self.original_mime_data.setData(format, mime_data.data(format))
                print("Backed up clipboard content")
        except Exception as e:
            print(f"Error backing up clipboard: {str(e)}")
    
    def restore_clipboard(self):
        """Restore the original clipboard content"""
        if self.original_mime_data:
            try:
                self.clipboard.setMimeData(self.original_mime_data)
                print("Restored clipboard content")
            except Exception as e:
                print(f"Error restoring clipboard: {str(e)}")
            finally:
                self.original_mime_data = None
    
    def set_clipboard_text(self, text):
        """Set the clipboard text using multiple methods for redundancy"""
        success = False
        
        try:
            self.clipboard.setText(text)
            print("Set clipboard using Qt")
            success = True
        except Exception as e:
            print(f"Qt clipboard set failed: {str(e)}")
        
        try:
            pyperclip.copy(text)
            print("Set clipboard using pyperclip")
            success = True
        except Exception as e:
            print(f"pyperclip set failed: {str(e)}")
        
        # Add a small delay to ensure clipboard is set
        time.sleep(0.1)
        
        return success
    
    def get_clipboard_content_type(self):
        """Get the type of content currently in the clipboard"""
        mime_data = self.clipboard.mimeData()
        content_types = []
        
        if mime_data.hasImage():
            content_types.append("image")
        if mime_data.hasText():
            content_types.append("text")
        if mime_data.hasHtml():
            content_types.append("html")
        if mime_data.hasUrls():
            content_types.append("urls")
        
        return content_types
    
    def paste_clipboard(self):
        """Simulate keyboard paste operation"""
        paste_success = False
        
        try:
            # First try using Ctrl+V method
            keyboard.press_and_release('ctrl+v')
            print("Used Ctrl+V method")
            paste_success = True
        except Exception as e1:
            print(f"Failed to use Ctrl+V: {str(e1)}")
            try:
                # Fallback to keyboard.write() method
                keyboard.write(self.clipboard.text())
                print("Used keyboard write method")
                paste_success = True
            except Exception as e2:
                print(f"Failed to use keyboard write: {str(e2)}")
        
        # Add a small delay after paste
        time.sleep(0.1)
        
        return paste_success 