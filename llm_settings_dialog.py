from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QCheckBox, QTextEdit, QDialogButtonBox,
    QGroupBox, QRadioButton
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence

class LLMSettingsDialog(QDialog):
    """Dialog for configuring LLM settings"""
    def __init__(self, parent=None, llm_processor=None, current_hotkey=None):
        super().__init__(parent)
        self.setWindowTitle("LLM Processing Settings")
        self.setModal(True)
        self.setMinimumWidth(500)
        
        self.llm_processor = llm_processor
        self.current_hotkey = current_hotkey
        self.recording_hotkey = False
        self.new_hotkey = None
        
        self._init_ui()
        self._load_current_settings()
    
    def _init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout()
        
        # Enable LLM processing checkbox
        self.enable_llm = QCheckBox("Enable LLM Processing")
        layout.addWidget(self.enable_llm)
        
        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout()
        self.push_to_talk_radio = QRadioButton("Push-to-Talk")
        self.toggle_mode_radio = QRadioButton("Toggle Mode")
        mode_layout.addWidget(self.push_to_talk_radio)
        mode_layout.addWidget(self.toggle_mode_radio)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Provider selection
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Gemini")
        provider_layout.addWidget(self.provider_combo)
        layout.addLayout(provider_layout)
        
        # API Key
        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API Key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self.api_key_edit)
        layout.addLayout(api_layout)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.llm_processor.get_available_models())
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Screenshot option
        self.include_screenshot = QCheckBox("Include screenshot with LLM requests")
        self.include_screenshot.setToolTip("When enabled, a screenshot of your screen will be captured and sent with each LLM request")
        layout.addWidget(self.include_screenshot)
        
        # Hotkey configuration
        hotkey_layout = QHBoxLayout()
        hotkey_layout.addWidget(QLabel("LLM Processing Hotkey:"))
        self.hotkey_edit = QLineEdit()
        self.hotkey_edit.setReadOnly(True)
        if self.current_hotkey:
            self.hotkey_edit.setText(self.current_hotkey)
        self.hotkey_edit.mousePressEvent = self._start_hotkey_recording
        hotkey_layout.addWidget(self.hotkey_edit)
        layout.addLayout(hotkey_layout)
        
        # Prompt template
        layout.addWidget(QLabel("Prompt Template:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Enter your prompt template. Use {text} as placeholder for the transcribed text.")
        self.prompt_edit.setMaximumHeight(100)
        layout.addWidget(self.prompt_edit)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
    
    def _load_current_settings(self):
        """Load current settings into the dialog"""
        if self.llm_processor:
            self.api_key_edit.setText(self.llm_processor.api_key)
            self.prompt_edit.setText(self.llm_processor.prompt_template)
            
            # Set mode
            if self.llm_processor.is_push_to_talk:
                self.push_to_talk_radio.setChecked(True)
            else:
                self.toggle_mode_radio.setChecked(True)
            
            # Set provider
            index = self.provider_combo.findText(self.llm_processor.provider.capitalize())
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
            
            # Set model
            index = self.model_combo.findText(self.llm_processor.model)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            
            # Set screenshot option
            self.include_screenshot.setChecked(self.llm_processor.include_screenshot)
    
    def _start_hotkey_recording(self, event):
        """Start recording a new hotkey"""
        self.recording_hotkey = True
        self.hotkey_edit.setText("Press keys...")
        self.hotkey_edit.setFocus()
    
    def keyPressEvent(self, event):
        """Handle key press events for hotkey recording"""
        if self.recording_hotkey:
            key = event.key()
            modifiers = event.modifiers()
            
            if key in (Qt.Key_Shift, Qt.Key_Control, Qt.Key_Alt, Qt.Key_Meta):
                return
            
            sequence = []
            if modifiers & Qt.ControlModifier:
                sequence.append("Ctrl")
            if modifiers & Qt.AltModifier:
                sequence.append("Alt")
            if modifiers & Qt.ShiftModifier:
                sequence.append("Shift")
            
            key_text = QKeySequence(key).toString()
            if key_text:
                sequence.append(key_text)
            
            if sequence:
                self.new_hotkey = "+".join(sequence)
                self.hotkey_edit.setText(self.new_hotkey)
            
            self.recording_hotkey = False
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def get_settings(self):
        """Get the configured settings"""
        return {
            'enabled': self.enable_llm.isChecked(),
            'provider': self.provider_combo.currentText().lower(),
            'api_key': self.api_key_edit.text(),
            'model': self.model_combo.currentText(),
            'hotkey': self.new_hotkey or self.current_hotkey,
            'prompt_template': self.prompt_edit.toPlainText(),
            'is_push_to_talk': self.push_to_talk_radio.isChecked(),
            'include_screenshot': self.include_screenshot.isChecked()
        } 