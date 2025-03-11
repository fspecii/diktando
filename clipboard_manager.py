#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import platform
import io
import pyperclip
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QMimeData, QBuffer
from PyQt5.QtGui import QImage, QPixmap

class ClipboardManager:
    """Manages clipboard operations for the application"""
    
    def __init__(self):
        """Initialize the clipboard manager"""
        self.original_text = None
        self.original_image = None
        self.original_mime_data = None
        self.has_image = False
        self.has_text = False
    
    def backup_clipboard(self):
        """Backup the current clipboard content including text and images"""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
        # Create a new mime data object to store a copy of the clipboard content
        self.original_mime_data = QMimeData()
        
        try:
            # Check for text
            if mime_data.hasText():
                self.original_text = mime_data.text()
                self.has_text = True
                self.original_mime_data.setText(self.original_text)
                print(f"Backed up clipboard text: {self.original_text[:30]}...")
            
            # Check for image
            if mime_data.hasImage():
                self.original_image = QImage(clipboard.image())
                self.has_image = True
                self.original_mime_data.setImageData(self.original_image)
                print("Backed up clipboard image")
            
            # Check for HTML
            if mime_data.hasHtml():
                html = mime_data.html()
                self.original_mime_data.setHtml(html)
                print("Backed up clipboard HTML")
            
            # Check for URLs
            if mime_data.hasUrls():
                urls = mime_data.urls()
                self.original_mime_data.setUrls(urls)
                print(f"Backed up clipboard URLs: {len(urls)} URLs")
            
            print("Clipboard backup complete")
            
        except Exception as e:
            print(f"Clipboard backup failed: {str(e)}")
            
            # Fallback to pyperclip for text only
            try:
                self.original_text = pyperclip.paste()
                self.has_text = True
                print("Backed up clipboard text using pyperclip")
            except Exception as e2:
                print(f"pyperclip backup failed: {str(e2)}")
    
    def restore_clipboard(self):
        """Restore the original clipboard content including text and images"""
        if not (self.has_text or self.has_image or self.original_mime_data):
            print("No clipboard content to restore")
            return
        
        # Add a small delay before restoration
        time.sleep(0.1)
        
        try:
            # Restore using the mime data if available
            if self.original_mime_data:
                QApplication.clipboard().setMimeData(self.original_mime_data)
                print("Restored clipboard using mime data")
            # Fallback methods if mime data restoration fails
            elif self.has_image:
                QApplication.clipboard().setImage(self.original_image)
                print("Restored clipboard image")
            elif self.has_text:
                QApplication.clipboard().setText(self.original_text)
                print("Restored clipboard text")
                
                # Additional fallback using pyperclip
                try:
                    pyperclip.copy(self.original_text)
                    print("Restored clipboard text using pyperclip")
                except Exception as e:
                    print(f"pyperclip restore failed: {str(e)}")
            
        except Exception as e:
            print(f"Clipboard restoration failed: {str(e)}")
            
            # Last resort fallback for text
            if self.has_text:
                try:
                    pyperclip.copy(self.original_text)
                    print("Restored clipboard text using pyperclip")
                except Exception as e2:
                    print(f"pyperclip restore failed: {str(e2)}")
        
        # Add a small delay after restoration
        time.sleep(0.1)
        
        # Reset the backup variables
        self.original_text = None
        self.original_image = None
        self.original_mime_data = None
        self.has_image = False
        self.has_text = False
    
    def set_clipboard_text(self, text):
        """Set the clipboard text using multiple methods for redundancy"""
        success = False
        
        try:
            QApplication.clipboard().setText(text)
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
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        
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
        """Simulate keyboard paste operation (Ctrl+V or Command+V)"""
        paste_success = False
        
        # Simulate paste based on platform
        if platform.system() == "Windows":
            try:
                import keyboard
                keyboard.press_and_release('ctrl+v')
                print("Paste command sent (Windows)")
                paste_success = True
            except Exception as e:
                print(f"Failed to simulate paste on Windows: {str(e)}")
        elif platform.system() == "Darwin":  # macOS
            try:
                from pynput import keyboard
                controller = keyboard.Controller()
                with controller.pressed(keyboard.Key.cmd):
                    controller.press('v')
                    controller.release('v')
                print("Paste command sent (macOS)")
                paste_success = True
            except Exception as e:
                print(f"Failed to simulate paste on macOS: {str(e)}")
        else:  # Linux
            try:
                from pynput import keyboard
                controller = keyboard.Controller()
                with controller.pressed(keyboard.Key.ctrl):
                    controller.press('v')
                    controller.release('v')
                print("Paste command sent (Linux)")
                paste_success = True
            except Exception as e:
                print(f"Failed to simulate paste on Linux: {str(e)}")
        
        # Add a small delay after paste
        time.sleep(0.1)
        
        return paste_success 