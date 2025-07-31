import sys
import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QFrame, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QObject
from PySide6.QtGui import QImage, QPixmap
from gtts import gTTS
from googletrans import Translator  # Added for translation
import pygame  # Used for playing the generated speech audio


# --- Text-to-Speech and Translation Worker ---
class TTSWorker(QObject) :
    """
    A worker that runs in a separate thread to handle text translation
    and speech conversion without blocking the main UI.
    """
    finished = Signal()
    error = Signal(str)

    def __init__(self, text, lang_code) :
        super().__init__()
        self.text = text
        self.lang_code = lang_code

    @Slot()
    def run(self) :
        """Translates the text and then generates and plays the speech."""
        if not self.text or self.text == "No sign detected" :
            self.finished.emit()
            return
        try :
            # Step 1: Translate the text
            translator = Translator()
            # Source language is English ('en')
            translation = translator.translate(self.text, src='en', dest=self.lang_code)
            translated_text = translation.text

            # Step 2: Create gTTS object with the translated text
            tts = gTTS(text=translated_text, lang=self.lang_code, slow=False)

            # Save to a temporary file
            speech_file = "temp_speech.mp3"
            tts.save(speech_file)

            # Step 3: Play the audio file using pygame
            pygame.mixer.init()
            pygame.mixer.music.load(speech_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() :
                pygame.time.Clock().tick(10)

        except Exception as e :
            self.error.emit(f"Translation/TTS Error: {e}")
        finally :
            # Clean up pygame mixer and the temp file
            if pygame.mixer.get_init() :
                pygame.mixer.music.stop()
                pygame.mixer.quit()
            if os.path.exists("temp_speech.mp3") :
                os.remove("temp_speech.mp3")
            self.finished.emit()


# --- Main Application Window ---
class MainWindow(QMainWindow) :
    """
    The main window for the Sign Language Detection application.
    It sets up the UI, manages widgets, and handles user interactions.
    """

    def __init__(self) :
        super().__init__()

        # --- Model and Class Names ---
        self.model = None
        self.class_names = None
        self.detected_text = ""
        self.model_path = "runs/detect/train/weights/best.pt"
        self.yaml_path = "data.yaml"

        # --- Window Configuration ---
        self.setWindowTitle("Indian Sign Language Detection with Translation & TTS")
        self.setGeometry(100, 100, 1200, 800)
        self.setup_ui()

        # --- Load Model and Classes ---
        self.load_resources()

        # --- Threads ---
        self.video_thread = None
        self.tts_thread = None
        self.tts_worker = None

    def setup_ui(self) :
        """Initializes the user interface, layout, and styles."""
        self.setStyleSheet("""
            QMainWindow { background-color: #2c3e50; }
            QLabel { color: #ecf0f1; font-size: 16px; }
            QPushButton {
                background-color: #3498db; color: white; font-size: 16px;
                font-weight: bold; border-radius: 15px; padding: 12px 24px;
                border: none;
            }
            QPushButton:hover { background-color: #2980b9; }
            QPushButton:pressed { background-color: #1f618d; }
            QPushButton:disabled { background-color: #566573; }
            QComboBox {
                border: 1px solid #34495e; border-radius: 5px; padding: 5px;
                background-color: #34495e; color: white; font-size: 14px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #34495e; color: white;
                selection-background-color: #3498db;
            }
            QFrame { border: 2px solid #34495e; border-radius: 10px; }
            #TitleLabel { font-size: 32px; font-weight: bold; color: #3498db; padding: 10px; }
            #ResultLabel {
                font-size: 24px; font-weight: bold; color: #2ecc71;
                background-color: #34495e; border-radius: 10px; padding: 15px;
            }
            #ImageLabel { background-color: #000; border-radius: 10px; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Indian Sign Language Detection")
        title_label.setObjectName("TitleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        video_frame = QFrame()
        video_layout = QVBoxLayout(video_frame)
        self.image_label = QLabel("Upload an image or start the webcam to begin.")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 600)
        video_layout.addWidget(self.image_label)
        content_layout.addWidget(video_frame, stretch=3)

        controls_frame = QFrame()
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setSpacing(15)
        controls_layout.setAlignment(Qt.AlignTop)

        self.btn_upload_image = QPushButton("Upload Image")
        self.btn_upload_image.clicked.connect(self.upload_image)
        controls_layout.addWidget(self.btn_upload_image)

        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_webcam.clicked.connect(self.toggle_webcam)
        controls_layout.addWidget(self.btn_webcam)

        # --- Language Selection and Speak Button ---
        tts_layout = QHBoxLayout()
        self.lang_combo = QComboBox()
        # Language Name -> gTTS language code
        self.languages = {
            "Hindi" : "hi", "Bengali" : "bn", "Tamil" : "ta",
            "Telugu" : "te", "Kannada" : "kn", "Gujarati" : "gu",
            "Marathi" : "mr", "English" : "en"
        }
        self.lang_combo.addItems(self.languages.keys())
        tts_layout.addWidget(self.lang_combo, stretch=2)

        self.btn_speak = QPushButton("Speak")
        self.btn_speak.clicked.connect(self.speak_result)
        self.btn_speak.setEnabled(False)  # Disabled by default
        tts_layout.addWidget(self.btn_speak, stretch=1)

        controls_layout.addLayout(tts_layout)

        self.result_label = QLabel("Result: No sign detected")
        self.result_label.setObjectName("ResultLabel")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        controls_layout.addWidget(self.result_label)

        controls_layout.addStretch()
        content_layout.addWidget(controls_frame, stretch=1)

    def load_resources(self) :
        """Loads the YOLO model and class names from files."""
        if not os.path.exists(self.model_path) or not os.path.exists(self.yaml_path) :
            self.update_result(f"Error: Model or YAML file not found.", is_error=True)
            self.btn_upload_image.setEnabled(False)
            self.btn_webcam.setEnabled(False)
            return
        try :
            with open(self.yaml_path, 'r') as file :
                data = yaml.safe_load(file)
            self.class_names = data['names']
            self.model = YOLO(self.model_path)
            self.update_result("Model loaded successfully.")
        except Exception as e :
            self.update_result(f"Failed to load resources: {e}", is_error=True)
            self.btn_upload_image.setEnabled(False)
            self.btn_webcam.setEnabled(False)

    @Slot()
    def upload_image(self) :
        """Opens a file dialog to upload and process an image."""
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
                self.update_result(f"Error processing image: {e}", is_error=True)

    @Slot()
    def toggle_webcam(self) :
        """Starts or stops the webcam video feed."""
        if self.model is None :
            self.update_result("Model not loaded. Cannot start webcam.", is_error=True)
            return

        if self.video_thread and self.video_thread.isRunning() :
            self.video_thread.stop()
            self.video_thread.wait()
            self.btn_webcam.setText("Start Webcam")
            self.image_label.setText("Webcam stopped.")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.btn_upload_image.setEnabled(True)
        else :
            self.video_thread = VideoThread(self.model, self.class_names)
            self.video_thread.frame_signal.connect(self.update_image)
            self.video_thread.result_signal.connect(self.update_result)
            self.video_thread.start()
            self.btn_webcam.setText("Stop Webcam")
            self.btn_upload_image.setEnabled(False)

    def detect_sign_language(self, frame) :
        """Performs sign language detection on a single frame."""
        results = self.model(frame)[0]
        detected_signs = []
        for box in results.boxes :
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5 :
                sign = self.class_names[cls_id]
                detected_signs.append(sign)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2 :]), (46, 204, 113), 3)
                label = f"{sign} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (xyxy[0], xyxy[1] - h - 15), (xyxy[0] + w, xyxy[1] - 5), (46, 204, 113), -1)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (44, 62, 80), 2)
        result_text = ", ".join(detected_signs) if detected_signs else "No sign detected"
        return frame, result_text

    @Slot(np.ndarray)
    def update_image(self, cv_img) :
        """Updates the image_label with a new frame from OpenCV."""
        qt_img = self.convert_cv_to_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @Slot(str)
    def update_result(self, text, is_error=False) :
        """Updates the result_label and enables/disables the speak button."""
        self.detected_text = text if not is_error else ""
        self.result_label.setText(f"Result: {text}")

        if is_error :
            self.result_label.setStyleSheet("color: #e74c3c;")  # Red
            self.btn_speak.setEnabled(False)
        else :
            self.result_label.setStyleSheet("color: #2ecc71;")  # Green
            is_valid_result = self.detected_text and self.detected_text != "No sign detected"
            self.btn_speak.setEnabled(is_valid_result)

    def convert_cv_to_qt(self, cv_img) :
        """Converts an OpenCV image (numpy array) to a QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(p)

    @Slot()
    def speak_result(self) :
        """Initiates the text-to-speech conversion in a separate thread."""
        selected_language_name = self.lang_combo.currentText()
        lang_code = self.languages[selected_language_name]

        self.btn_speak.setEnabled(False)
        self.btn_speak.setText("Speaking...")

        # Setup and run TTS in a separate thread
        self.tts_thread = QThread()
        self.tts_worker = TTSWorker(self.detected_text, lang_code)
        self.tts_worker.moveToThread(self.tts_thread)

        self.tts_thread.started.connect(self.tts_worker.run)
        self.tts_worker.finished.connect(self.tts_thread.quit)
        self.tts_worker.finished.connect(self.on_tts_finished)
        self.tts_worker.error.connect(self.on_tts_error)

        self.tts_thread.start()

    def on_tts_finished(self) :
        """Cleans up after TTS is done."""
        self.btn_speak.setText("Speak")
        self.btn_speak.setEnabled(True)
        if self.tts_thread :
            self.tts_thread.deleteLater()
            self.tts_worker.deleteLater()
            self.tts_thread = None
            self.tts_worker = None

    def on_tts_error(self, error_message) :
        """Handles errors from the TTS worker."""
        self.update_result(error_message, is_error=True)
        self.on_tts_finished()

    def closeEvent(self, event) :
        """Ensures threads are stopped when the window closes."""
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

        while self._is_running and cap.isOpened() :
            ret, frame = cap.read()
            if not ret :
                break

            processed_frame, result_text = self.detect_sign_language(frame)
            self.frame_signal.emit(processed_frame)
            self.result_signal.emit(result_text)
            self.msleep(30)
        cap.release()

    def detect_sign_language(self, frame) :
        results = self.model(frame)[0]
        detected_signs = []
        for box in results.boxes :
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.5 :
                sign = self.class_names[cls_id]
                detected_signs.append(sign)
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2 :]), (46, 204, 113), 3)
                label = f"{sign} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (xyxy[0], xyxy[1] - h - 15), (xyxy[0] + w, xyxy[1] - 5), (46, 204, 113), -1)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (44, 62, 80), 2)
        result_text = ", ".join(detected_signs) if detected_signs else "No sign detected"
        return frame, result_text

    def stop(self) :
        self._is_running = False


# --- Entry Point ---
if __name__ == "__main__" :
    # Required packages:
    # pip install PySide6 opencv-python numpy ultralytics pyyaml gTTS pygame googletrans==4.0.0-rc1

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

