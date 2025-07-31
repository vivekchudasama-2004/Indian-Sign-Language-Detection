# main_app.py

import sys
import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QComboBox, QStatusBar
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon, QPainter
from gtts import gTTS
from googletrans import Translator
import pygame

# --- SVG Icon Definitions ---
# Using SVG for icons makes the app self-contained and scalable.
ICON_UPLOAD = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
"""

ICON_WEBCAM_START = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
"""

ICON_WEBCAM_STOP = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="1" y1="1" x2="23" y2="23" /><path d="M21 21H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h3m3-3h6l2 3h4a2 2 0 0 1 2 2v9.34m-7.72-2.06a4 4 0 1 1-5.56-5.56" /></svg>
"""

ICON_SPEAK = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/></svg>
"""


# --- Text-to-Speech and Translation Worker ---
class TTSWorker(QObject) :
    finished = Signal()
    error = Signal(str)

    def __init__(self, text, lang_code) :
        super().__init__()
        self.text = text
        self.lang_code = lang_code

    @Slot()
    def run(self) :
        if not self.text or self.text == "No sign detected" :
            self.finished.emit()
            return
        try :
            translator = Translator()
            translation = translator.translate(self.text, src='en', dest=self.lang_code)
            translated_text = translation.text

            tts = gTTS(text=translated_text, lang=self.lang_code, slow=False)
            speech_file = "temp_speech.mp3"
            tts.save(speech_file)

            pygame.mixer.init()
            pygame.mixer.music.load(speech_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() :
                pygame.time.Clock().tick(10)

        except Exception as e :
            self.error.emit(f"Translation/TTS Error: {e}")
        finally :
            if pygame.mixer.get_init() :
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            if os.path.exists("temp_speech.mp3") :
                os.remove("temp_speech.mp3")
            self.finished.emit()


# --- Main Application Window ---
class MainWindow(QMainWindow) :
    def __init__(self) :
        super().__init__()
        self.model = None
        self.class_names = None
        self.detected_text = ""
        self.model_path = "runs/detect/train/weights/best.pt"
        self.yaml_path = "data.yaml"
        self.video_thread = None
        self.tts_thread = None
        self.tts_worker = None

        self.setWindowTitle("Indian Sign Language Detection")
        self.setGeometry(100, 100, 1280, 720)
        self.setup_ui()
        self.load_resources()

    def get_icon(self, svg_data) :
        """Creates a QIcon from SVG data."""
        pixmap = QPixmap(QSize(256, 256))
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(0, 0, QPixmap.fromImage(QImage.fromData(svg_data.encode('utf-8'))))
        painter.end()
        return QIcon(pixmap)

    def setup_ui(self) :
        self.setStyleSheet("""
            QMainWindow { background-color: #2c3e50; }
            QFrame#ControlFrame { background-color: #34495e; border-radius: 10px; }
            QLabel { color: #ecf0f1; font-size: 16px; font-family: 'Segoe UI'; }
            QPushButton {
                background-color: #3498db; color: white; font-size: 16px;
                font-weight: bold; border-radius: 8px; padding: 10px;
                border: none; text-align: left; padding-left: 20px;
            }
            QPushButton:hover { background-color: #4ea8e1; }
            QPushButton:pressed { background-color: #2980b9; }
            QPushButton:disabled { background-color: #566573; color: #95a5a6; }
            QComboBox {
                border: 1px solid #2c3e50; border-radius: 8px; padding: 8px;
                background-color: #2c3e50; color: white; font-size: 14px;
            }
            QComboBox::drop-down { border: none; subcontrol-origin: padding; subcontrol-position: top right; width: 20px; }
            QComboBox QAbstractItemView {
                background-color: #34495e; color: white; border-radius: 5px;
                selection-background-color: #3498db;
            }
            #TitleLabel { font-size: 28px; font-weight: bold; color: #ecf0f1; }
            #ResultLabel {
                font-size: 22px; font-weight: bold; color: #ecf0f1;
                background-color: #2c3e50; border-radius: 8px; padding: 15px;
            }
            #ImageLabel { background-color: #000; border-radius: 10px; }
            QStatusBar { color: #bdc3c7; font-size: 12px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Left Panel (Video Display) ---
        self.image_label = QLabel("Upload an image or start the webcam to begin.")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        main_layout.addWidget(self.image_label, stretch=3)

        # --- Right Panel (Controls) ---
        controls_frame = QFrame()
        controls_frame.setObjectName("ControlFrame")
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(20)
        controls_layout.setAlignment(Qt.AlignTop)

        title_label = QLabel("Sign Language AI")
        title_label.setObjectName("TitleLabel")
        controls_layout.addWidget(title_label)

        self.btn_upload_image = QPushButton("Upload Image")
        self.btn_upload_image.setIcon(self.get_icon(ICON_UPLOAD))
        self.btn_upload_image.setIconSize(QSize(24, 24))
        self.btn_upload_image.clicked.connect(self.upload_image)
        controls_layout.addWidget(self.btn_upload_image)

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_webcam.setIcon(self.get_icon(ICON_WEBCAM_START))
        self.btn_webcam.setIconSize(QSize(24, 24))
        self.btn_webcam.clicked.connect(self.toggle_webcam)
        controls_layout.addWidget(self.btn_webcam)

        controls_layout.addStretch(1)

        # --- Language Selection and Speak Button ---
        controls_layout.addWidget(QLabel("Translate & Speak"))
        self.lang_combo = QComboBox()
        self.languages = {
            "Hindi" : "hi", "Bengali" : "bn", "Tamil" : "ta",
            "Telugu" : "te", "Kannada" : "kn", "Gujarati" : "gu",
            "Marathi" : "mr", "English" : "en"
        }
        self.lang_combo.addItems(self.languages.keys())
        controls_layout.addWidget(self.lang_combo)

        self.btn_speak = QPushButton("Speak")
        self.btn_speak.setIcon(self.get_icon(ICON_SPEAK))
        self.btn_speak.setIconSize(QSize(24, 24))
        self.btn_speak.clicked.connect(self.speak_result)
        self.btn_speak.setEnabled(False)
        controls_layout.addWidget(self.btn_speak)

        controls_layout.addStretch(1)

        # --- Result Display ---
        controls_layout.addWidget(QLabel("Detection Result"))
        self.result_label = QLabel("...")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        controls_layout.addWidget(self.result_label)

        controls_layout.addStretch(5)
        main_layout.addWidget(controls_frame, stretch=1)

        # --- Status Bar ---
        self.setStatusBar(QStatusBar(self))

    def load_resources(self) :
        self.statusBar().showMessage("Loading AI model...")
        if not os.path.exists(self.model_path) or not os.path.exists(self.yaml_path) :
            self.update_result("Model not found", is_error=True)
            self.statusBar().showMessage("Error: Model or YAML file not found.")
            self.btn_upload_image.setEnabled(False)
            self.btn_webcam.setEnabled(False)
            return
        try :
            with open(self.yaml_path, 'r') as file :
                data = yaml.safe_load(file)
            self.class_names = data['names']
            self.model = YOLO(self.model_path)
            self.update_result("Ready", is_error=False)
            self.statusBar().showMessage("Model loaded successfully. Ready to detect.")
        except Exception as e :
            self.update_result("Load Error", is_error=True)
            self.statusBar().showMessage(f"Failed to load resources: {e}")
            self.btn_upload_image.setEnabled(False)
            self.btn_webcam.setEnabled(False)

    @Slot()
    def upload_image(self) :
        if self.video_thread and self.video_thread.isRunning() :
            self.toggle_webcam()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path and self.model :
            try :
                frame = cv2.imread(file_path)
                processed_frame, result_text = self.detect_sign_language(frame)
                self.update_image(processed_frame)
                self.update_result(result_text)
            except Exception as e :
                self.update_result(f"Image Error", is_error=True)
                self.statusBar().showMessage(f"Error processing image: {e}")

    @Slot()
    def toggle_webcam(self) :
        if not self.model : return
        if self.video_thread and self.video_thread.isRunning() :
            self.video_thread.stop()
            self.video_thread.wait()
            self.btn_webcam.setText("Start Webcam")
            self.btn_webcam.setIcon(self.get_icon(ICON_WEBCAM_START))
            self.image_label.setText("Webcam stopped.")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.btn_upload_image.setEnabled(True)
        else :
            self.video_thread = VideoThread(self.model, self.class_names)
            self.video_thread.frame_signal.connect(self.update_image)
            self.video_thread.result_signal.connect(self.update_result)
            self.video_thread.start()
            self.btn_webcam.setText("Stop Webcam")
            self.btn_webcam.setIcon(self.get_icon(ICON_WEBCAM_STOP))
            self.btn_upload_image.setEnabled(False)

    def detect_sign_language(self, frame) :
        results = self.model(frame)[0]
        detected_signs = [self.class_names[int(box.cls[0])] for box in results.boxes if float(box.conf[0]) > 0.5]

        for box in results.boxes :
            if float(box.conf[0]) > 0.5 :
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2 :]), (46, 204, 113), 3)
                label = f"{self.class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (xyxy[0], xyxy[1] - h - 15), (xyxy[0] + w, xyxy[1] - 5), (46, 204, 113), -1)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (44, 62, 80), 2)

        return frame, ", ".join(detected_signs) if detected_signs else "No sign detected"

    @Slot(np.ndarray)
    def update_image(self, cv_img) :
        qt_img = self.convert_cv_to_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @Slot(str)
    def update_result(self, text, is_error=False) :
        self.detected_text = "" if is_error else text
        self.result_label.setText(text)

        if is_error :
            self.result_label.setStyleSheet(
                "background-color: #e74c3c; color: white; border-radius: 8px; padding: 15px;")
            self.btn_speak.setEnabled(False)
        else :
            self.result_label.setStyleSheet(
                "background-color: #27ae60; color: white; border-radius: 8px; padding: 15px;")
            is_valid_result = self.detected_text and self.detected_text != "No sign detected"
            self.btn_speak.setEnabled(is_valid_result)

    def convert_cv_to_qt(self, cv_img) :
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(
            convert_to_Qt_format.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @Slot()
    def speak_result(self) :
        lang_code = self.languages[self.lang_combo.currentText()]
        self.btn_speak.setEnabled(False)
        self.btn_speak.setText("Speaking...")
        self.statusBar().showMessage(f"Translating '{self.detected_text}' to {self.lang_combo.currentText()}...")

        self.tts_thread = QThread()
        self.tts_worker = TTSWorker(self.detected_text, lang_code)
        self.tts_worker.moveToThread(self.tts_thread)
        self.tts_thread.started.connect(self.tts_worker.run)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.error.connect(self.on_tts_error)
        self.tts_thread.start()

    def on_tts_finished(self) :
        self.btn_speak.setText("Speak")
        self.btn_speak.setEnabled(True)
        self.statusBar().showMessage("Ready.", 3000)
        if self.tts_thread :
            self.tts_thread.quit()
            self.tts_thread.wait()

    def on_tts_error(self, error_message) :
        self.statusBar().showMessage(error_message, 5000)
        self.on_tts_finished()

    def closeEvent(self, event) :
        if self.video_thread and self.video_thread.isRunning() :
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


# --- Video Processing Thread ---
class VideoThread(QThread) :
    frame_signal = Signal(np.ndarray)
    result_signal = Signal(str)

    def __init__(self, model, class_names) :
        super().__init__()
        self.model = model
        self.class_names = class_names
        self._is_running = True

    def run(self) :
        cap = cv2.VideoCapture(0)
        if not cap.isOpened() :
            self.result_signal.emit("Error: Failed to access webcam.")
            return
        while self._is_running :
            ret, frame = cap.read()
            if not ret : break
            processed_frame, result_text = self.detect_sign_language(frame)
            self.frame_signal.emit(processed_frame)
            self.result_signal.emit(result_text)
            self.msleep(30)
        cap.release()

    def detect_sign_language(self, frame) :
        results = self.model(frame)[0]
        detected_signs = [self.class_names[int(box.cls[0])] for box in results.boxes if float(box.conf[0]) > 0.5]
        for box in results.boxes :
            if float(box.conf[0]) > 0.5 :
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2 :]), (46, 204, 113), 3)
                label = f"{self.class_names[int(box.cls[0])]} {float(box.conf[0]):.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (xyxy[0], xyxy[1] - h - 15), (xyxy[0] + w, xyxy[1] - 5), (46, 204, 113), -1)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (44, 62, 80), 2)
        return frame, ", ".join(detected_signs) if detected_signs else "No sign detected"

    def stop(self) :
        self._is_running = False


# --- Entry Point ---
if __name__ == "__main__" :
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
