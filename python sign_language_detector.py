import cv2
import numpy as np
from inference import get_model
import supervision as sv
import time
from collections import defaultdict, deque
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading

class SignLanguageDetector:
    def __init__(self):
        # Initialize the model - replace with your actual model ID
        self.model_id = "indian-sign-language-fkx1f/2"  # Update this
        self.api_key="rf_mSp8g0H7X1WLwGCeeHxtzK2S9w62"
        self.model = None
        self.cap = None
        self.is_detecting = False
        self.is_running = False
       
        
        # Detection statistics
        self.detection_stats = {
            'total_detections': 0,
            'unique_words': set(),
            'confidence_sum': 0,
            'frame_count': 0
        }
        
        # Word history (last 20 detections)
        self.word_history = deque(maxlen=20)
        
        # Sign language classes from your dataset
        self.classes = [
            'Bad', 'Brother', 'Father', 'Food', 'Friend', 'Good', 'Hello', 'Help', 
            'House', 'I', 'Indian', 'Loud', 'Mummy', 'Namaste', 'Name', 'No', 
            'Place', 'Please', 'Quiet', 'Sleeping', 'Sorry', 'Strong', 'Thank-you', 
            'Time', 'Today', 'Water', 'What', 'Yes', 'Your', 'language', 'sign', 'you'
        ]
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Indian Sign Language Detector")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), foreground='white', background='#2c3e50')
        style.configure('Heading.TLabel', font=('Arial', 14, 'bold'), foreground='white', background='#34495e')
        style.configure('Info.TLabel', font=('Arial', 12), foreground='white', background='#34495e')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="🤟 Indian Sign Language Detector", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#2c3e50')
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Video and controls
        left_panel = tk.Frame(content_frame, bg='#34495e', relief='raised', bd=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Video frame
        video_frame = tk.Frame(left_panel, bg='#34495e')
        video_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.video_label = tk.Label(video_frame, bg='black', text='Video Feed', fg='white', font=('Arial', 20))
        self.video_label.pack(fill='both', expand=True)
        
        # Controls frame
        controls_frame = tk.Frame(left_panel, bg='#34495e')
        controls_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Buttons
        self.start_btn = tk.Button(controls_frame, text='Start Camera', command=self.start_camera,
                                  bg='#27ae60', fg='white', font=('Arial', 12, 'bold'),
                                  padx=20, pady=10, relief='raised')
        self.start_btn.pack(side='left', padx=(0, 10))
        
        self.detect_btn = tk.Button(controls_frame, text='Start Detection', command=self.toggle_detection,
                                   bg='#3498db', fg='white', font=('Arial', 12, 'bold'),
                                   padx=20, pady=10, relief='raised', state='disabled')
        self.detect_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(controls_frame, text='Stop', command=self.stop_everything,
                                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                 padx=20, pady=10, relief='raised', state='disabled')
        self.stop_btn.pack(side='left')
        
        # Status label
        self.status_label = tk.Label(controls_frame, text='Ready to start', bg='#34495e', fg='#2ecc71',
                                    font=('Arial', 12, 'bold'))
        self.status_label.pack(side='right')
        
        # Right panel - Results and statistics
        right_panel = tk.Frame(content_frame, bg='#34495e', relief='raised', bd=2)
        right_panel.pack(side='right', fill='both', padx=(10, 0))
        right_panel.configure(width=400)
        
        # Results section
        results_frame = tk.Frame(right_panel, bg='#34495e')
        results_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        ttk.Label(results_frame, text="🎯 Current Detections", style='Heading.TLabel').pack(anchor='w')
        
        # Current detections
        self.detections_text = tk.Text(results_frame, height=8, bg='#2c3e50', fg='white',
                                      font=('Arial', 11), relief='sunken', bd=2)
        self.detections_text.pack(fill='x', pady=(10, 20))
        
        ttk.Label(results_frame, text="📝 Recent Words", style='Heading.TLabel').pack(anchor='w')
        
        # Word history
        self.history_text = tk.Text(results_frame, height=10, bg='#2c3e50', fg='white',
                                   font=('Arial', 10), relief='sunken', bd=2)
        self.history_text.pack(fill='both', expand=True, pady=(10, 20))
        
        # Statistics section
        stats_frame = tk.Frame(right_panel, bg='#34495e')
        stats_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        ttk.Label(stats_frame, text="📊 Statistics", style='Heading.TLabel').pack(anchor='w')
        
        stats_grid = tk.Frame(stats_frame, bg='#34495e')
        stats_grid.pack(fill='x', pady=10)
        
        # Statistics labels
        self.total_label = ttk.Label(stats_grid, text="Total: 0", style='Info.TLabel')
        self.total_label.grid(row=0, column=0, sticky='w', padx=(0, 20))
        
        self.unique_label = ttk.Label(stats_grid, text="Unique: 0", style='Info.TLabel')
        self.unique_label.grid(row=0, column=1, sticky='w')
        
        self.confidence_label = ttk.Label(stats_grid, text="Avg Confidence: 0%", style='Info.TLabel')
        self.confidence_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=(5, 0))
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = get_model(model_id=self.model_id)
            self.update_status("Model loaded successfully", "success")
            return True
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", "error")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            return False
    
    def start_camera(self):
        """Start the camera feed"""
        try:
            # Load model first
            if not self.load_model():
                return
                
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.is_running = True
            self.start_btn.configure(state='disabled', text='Camera Started ✓')
            self.detect_btn.configure(state='normal')
            self.stop_btn.configure(state='normal')
            self.update_status("Camera started successfully", "success")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
        except Exception as e:
            self.update_status(f"Camera error: {str(e)}", "error")
            messagebox.showerror("Camera Error", f"Failed to start camera: {str(e)}")
    
    def video_loop(self):
        """Main video processing loop"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Perform detection if enabled
            if self.is_detecting:
                annotated_frame = self.detect_signs(frame)
            else:
                annotated_frame = frame
            
            # Convert frame for display
            self.display_frame(annotated_frame)
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)
    
    def detect_signs(self, frame):
        """Detect sign language in the frame"""
        try:
            # Run inference
            results = self.model.infer(frame)[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_inference(results)
            
            # Filter detections by confidence threshold
            confidence_threshold = 0.5
            high_confidence_mask = detections.confidence >= confidence_threshold
            detections = detections[high_confidence_mask]
            
            # Update statistics and results
            if len(detections) > 0:
                self.update_detection_stats(detections)
                self.update_detection_display(detections)
            
            # Annotate frame
            annotated_frame = self.box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections)
            
            return annotated_frame
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def update_detection_stats(self, detections):
        """Update detection statistics"""
        num_detections = len(detections)
        self.detection_stats['total_detections'] += num_detections
        self.detection_stats['frame_count'] += 1
        
        for i in range(num_detections):
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]
            class_name = self.classes[class_id] if class_id < len(self.classes) else "Unknown"
            
            self.detection_stats['confidence_sum'] += confidence
            self.detection_stats['unique_words'].add(class_name)
            
            # Add to word history if confidence is high
            if confidence > 0.7:
                timestamp = time.strftime("%H:%M:%S")
                self.word_history.append({
                    'word': class_name,
                    'confidence': confidence,
                    'timestamp': timestamp
                })
        
        # Update statistics display
        self.root.after(0, self.update_stats_display)
    
    def update_detection_display(self, detections):
        """Update the current detections display"""
        def update_ui():
            self.detections_text.delete(1.0, tk.END)
            
            if len(detections) == 0:
                self.detections_text.insert(tk.END, "No signs detected")
                return
            
            for i in range(len(detections)):
                confidence = detections.confidence[i]
                class_id = detections.class_id[i]
                class_name = self.classes[class_id] if class_id < len(self.classes) else "Unknown"
                
                detection_text = f"🤟 {class_name}\n   Confidence: {confidence:.1%}\n\n"
                self.detections_text.insert(tk.END, detection_text)
            
            # Update word history display
            self.update_history_display()
        
        self.root.after(0, update_ui)
    
    def update_history_display(self):
        """Update the word history display"""
        self.history_text.delete(1.0, tk.END)
        
        if not self.word_history:
            self.history_text.insert(tk.END, "Word history will appear here...")
            return
        
        for item in reversed(list(self.word_history)):
            history_text = f"[{item['timestamp']}] {item['word']} ({item['confidence']:.1%})\n"
            self.history_text.insert(tk.END, history_text)
    
    def update_stats_display(self):
        """Update the statistics display"""
        total = self.detection_stats['total_detections']
        unique = len(self.detection_stats['unique_words'])
        avg_confidence = (self.detection_stats['confidence_sum'] / total * 100) if total > 0 else 0
        
        self.total_label.configure(text=f"Total: {total}")
        self.unique_label.configure(text=f"Unique: {unique}")
        self.confidence_label.configure(text=f"Avg Confidence: {avg_confidence:.1f}%")
    
    def display_frame(self, frame):
        """Display frame in the GUI"""
        # Resize frame to fit display
        display_frame = cv2.resize(frame, (640, 480))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        def update_label():
            self.video_label.configure(image=photo)
            self.video_label.image = photo  # Keep a reference
        
        self.root.after(0, update_label)
    
    def toggle_detection(self):
        """Toggle sign detection on/off"""
        if not self.is_detecting:
            self.start_detection()
        else:
            self.stop_detection()
    
    def start_detection(self):
        """Start sign detection"""
        self.is_detecting = True
        self.detect_btn.configure(text='Stop Detection', bg='#e74c3c')
        self.update_status("🔍 Detecting signs...", "detecting")
    
    def stop_detection(self):
        """Stop sign detection"""
        self.is_detecting = False
        self.detect_btn.configure(text='Start Detection', bg='#3498db')
        self.update_status("Detection stopped", "ready")
    
    def stop_everything(self):
        """Stop camera and detection"""
        self.is_running = False
        self.is_detecting = False
        
        if self.cap:
            self.cap.release()
        
        self.start_btn.configure(state='normal', text='Start Camera')
        self.detect_btn.configure(state='disabled', text='Start Detection', bg='#3498db')
        self.stop_btn.configure(state='disabled')
        
        # Clear video display
        self.video_label.configure(image='', text='Video Feed')
        self.video_label.image = None
        
        self.update_status("Stopped", "ready")
    
    def update_status(self, message, status_type):
        """Update status message"""
        colors = {
            "ready": "#2ecc71",
            "detecting": "#3498db",
            "success": "#2ecc71",
            "error": "#e74c3c"
        }
        
        def update_ui():
            self.status_label.configure(text=message, fg=colors.get(status_type, "#2ecc71"))
        
        self.root.after(0, update_ui)
    
    def run(self):
        """Run the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        self.stop_everything()
        self.root.destroy()


# Alternative version using OpenCV window (simpler)
class SimpleSignLanguageDetector:
    def __init__(self, model_id):
        self.model_id = model_id
        self.model = None
        self.classes = [
            'Bad', 'Brother', 'Father', 'Food', 'Friend', 'Good', 'Hello', 'Help', 
            'House', 'I', 'Indian', 'Loud', 'Mummy', 'Namaste', 'Name', 'No', 
            'Place', 'Please', 'Quiet', 'Sleeping', 'Sorry', 'Strong', 'Thank-you', 
            'Time', 'Today', 'Water', 'What', 'Yes', 'Your', 'language', 'sign', 'you'
        ]
        
        # Initialize supervision annotators
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            print("Loading model...")
            self.model = get_model(model_id=self.model_id)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)
    
    def run_detection(self):
        """Run real-time detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting real-time detection...")
        print("Press 'q' to quit, 's' to save current frame")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # Perform detection every few frames to improve performance
            if frame_count % 5 == 0:
                try:
                    # Run inference
                    results = self.model.infer(frame)[0]
                    
                    # Convert to supervision format
                    detections = sv.Detections.from_inference(results)
                    
                    # Filter by confidence
                    confidence_threshold = 0.5
                    high_confidence_mask = detections.confidence >= confidence_threshold
                    detections = detections[high_confidence_mask]
                    
                    if len(detections) > 0:
                        detection_count += len(detections)
                        
                        # Print detections to console
                        for i in range(len(detections)):
                            confidence = detections.confidence[i]
                            class_id = detections.class_id[i]
                            class_name = self.classes[class_id] if class_id < len(self.classes) else "Unknown"
                            print(f"Detected: {class_name} ({confidence:.2f})")
                    
                    # Annotate frame
                    frame = self.box_annotator.annotate(scene=frame, detections=detections)
                    frame = self.label_annotator.annotate(scene=frame, detections=detections)
                    
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Add info text to frame
            cv2.putText(frame, f"Detections: {detection_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Indian Sign Language Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'detection_frame_{int(time.time())}.jpg', frame)
                print("Frame saved!")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Total detections: {detection_count}")


# Usage instructions and main execution
if __name__ == "__main__":
    print("Indian Sign Language Detector")
    print("="*50)
    
    # Replace with your actual Roboflow model ID
    MODEL_ID = "indian-sign-language-fkx1f/2"
    
    print("Choose detection mode:")
    print("1. GUI Mode (Recommended)")
    print("2. Simple OpenCV Mode")
    
    choice = '2'
    
    if choice == "1":
        print("\nStarting GUI mode...")
        detector = SignLanguageDetector()
        detector.model_id = MODEL_ID
        detector.run()
    elif choice == "2":
        print("\nStarting simple mode...")
        detector = SimpleSignLanguageDetector(MODEL_ID)
        detector.run_detection()
    else:
        print("Invalid choice. Starting GUI mode by default...")
        detector = SignLanguageDetector()
        detector.model_id = MODEL_ID
        detector.run()