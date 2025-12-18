import cv2
import numpy as np
import mediapipe as mp
import joblib
import json
import os
import time
from datetime import datetime
import pandas as pd
from collections import deque, Counter
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from flask import Flask
from threading import Thread

# Flask app for web interface
app = Flask(__name__)

# Global variable to store current prediction
current_letter = "None"

@app.route('/')
def index():
    return current_letter

class ASLHandSignDetector:
    def __init__(self, model_dir="trained_models"):
        """
        Real-time ASL hand sign detector with web server support
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.pca = None
        self.metadata = None
        self.label_classes = []
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Detection settings
        self.prediction_history = deque(maxlen=15)
        self.confidence_threshold = 0.4
        self.diversity_requirement = False
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.prediction_log = []
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.thickness = 2
        
        print("ASL Hand Sign Detector v3.0 - REAL ASL MODEL with Web Interface")
    
    def find_latest_model_files(self):
        """Find the most recent model files"""
        print(f"Searching for models in: {self.model_dir}")
        
        if not os.path.exists(self.model_dir):
            print(f"Model directory not found: {self.model_dir}")
            return None, None, None, None
        
        all_files = os.listdir(self.model_dir)
        
        model_files = [f for f in all_files if f.endswith('.joblib') and 
                      ('model' in f.lower() and 'scaler' not in f and 'pca' not in f)]
        scaler_files = [f for f in all_files if 'scaler' in f.lower() and f.endswith('.joblib')]
        metadata_files = [f for f in all_files if f.startswith('model_metadata') and f.endswith('.json')]
        pca_files = [f for f in all_files if 'pca' in f.lower() and f.endswith('.joblib')]
        
        if not model_files:
            print("No model files found!")
            return None, None, None, None
        
        def get_most_recent(file_list):
            if not file_list:
                return None
            
            files_with_time = []
            for f in file_list:
                full_path = os.path.join(self.model_dir, f)
                mod_time = os.path.getmtime(full_path)
                files_with_time.append((f, mod_time))
            
            files_with_time.sort(key=lambda x: x[1], reverse=True)
            return files_with_time[0][0]
        
        model_file = get_most_recent(model_files)
        scaler_file = get_most_recent(scaler_files)
        metadata_file = get_most_recent(metadata_files)
        pca_file = get_most_recent(pca_files)
        
        return model_file, scaler_file, metadata_file, pca_file
    
    def load_trained_model(self, model_name=None):
        """Load model with PCA transformer support"""
        print("Loading REAL ASL model with PCA transformer...")
        
        model_file, scaler_file, metadata_file, pca_file = self.find_latest_model_files()
        
        if not model_file:
            print("No model files found!")
            return False
        
        # Load metadata
        if metadata_file:
            metadata_path = os.path.join(self.model_dir, metadata_file)
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                
                if self.metadata.get('real_asl_model', False):
                    self.confidence_threshold = 0.15
                    self.diversity_requirement = False
                
            except Exception as e:
                self.metadata = {}
        else:
            self.metadata = {}
        
        # Load model
        model_path = os.path.join(self.model_dir, model_file)
        try:
            self.model = joblib.load(model_path)
            
            if hasattr(self.model, 'classes_'):
                actual_classes = list(self.model.classes_)
            else:
                actual_classes = self.metadata.get('label_classes', [])
            
            self.label_classes = actual_classes
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
        
        # Load scaler
        if scaler_file:
            scaler_path = os.path.join(self.model_dir, scaler_file)
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception as e:
                self.scaler = None
        else:
            self.scaler = None
        
        # Load PCA transformer
        if pca_file:
            pca_path = os.path.join(self.model_dir, pca_file)
            try:
                self.pca = joblib.load(pca_path)
            except Exception as e:
                self.pca = None
        else:
            self.pca = None
        
        return True
    
    def get_camera_id(self):
        """Get camera ID from user input"""
        while True:
            try:
                camera_input = input("Enter camera ID (usually 0 for default camera): ").strip()
                camera_id = int(camera_input)
                
                test_cap = cv2.VideoCapture(camera_id)
                
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    if ret:
                        test_cap.release()
                        return camera_id
                    else:
                        test_cap.release()
                
                retry = input("Try another camera ID? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
                    
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                return None
    
    def extract_hand_landmarks(self, image):
        """Extract hand landmarks using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        results = self.hands.process(rgb_image)
        
        landmarks_data = []
        hand_info = []
        
        if results.multi_hand_landmarks:
            for i, (hand_landmarks, handedness) in enumerate(zip(
                results.multi_hand_landmarks, 
                results.multi_handedness
            )):
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                landmarks_data.append(landmarks)
                hand_info.append({
                    'hand_index': i,
                    'handedness': handedness.classification[0].label,
                    'confidence': handedness.classification[0].score,
                    'landmarks': hand_landmarks
                })
        
        return landmarks_data, hand_info, results
    
    def create_feature_vector(self, landmarks_list):
        """Create feature vector with proper PCA transformation pipeline"""
        if not landmarks_list:
            return None
        
        landmarks = landmarks_list[0]
        expected_features = self.metadata.get('original_features', 63)
        
        if len(landmarks) != expected_features:
            if len(landmarks) < expected_features:
                landmarks.extend([0.0] * (expected_features - len(landmarks)))
            else:
                landmarks = landmarks[:expected_features]
        
        landmarks_array = np.array(landmarks).reshape(1, -1)
        
        if self.scaler:
            try:
                landmarks_scaled = self.scaler.transform(landmarks_array)
            except Exception as e:
                return None
        else:
            return None
        
        if self.pca:
            try:
                landmarks_pca = self.pca.transform(landmarks_scaled)
                return landmarks_pca
            except Exception as e:
                return None
        else:
            return None
    
    def predict_sign(self, feature_vector):
        """Make prediction with real ASL model"""
        if feature_vector is None or self.model is None:
            return None, 0.0
        
        try:
            prediction = self.model.predict(feature_vector)[0]
            
            confidence = 0.0
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_vector)[0]
                confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            return None, 0.0
    
    def smooth_predictions(self, prediction, confidence):
        """Simplified smoothing for real ASL model"""
        if prediction is None:
            return None, 0.0
        
        self.prediction_history.append((prediction, confidence))
        
        recent_above_threshold = [p for p, c in self.prediction_history if c > self.confidence_threshold]
        
        if len(recent_above_threshold) < 3:
            return None, 0.0
        
        prediction_counts = Counter(recent_above_threshold)
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        
        confidences = [c for p, c in self.prediction_history if p == most_common_prediction]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        if prediction_counts[most_common_prediction] >= 2:
            return most_common_prediction, avg_confidence
        
        return None, 0.0
    
    def update_web_data(self, prediction):
        """Update the global prediction data for web interface"""
        global current_letter
        current_letter = str(prediction) if prediction is not None else "None"
    
    def draw_predictions(self, image, prediction, confidence, hand_info):
        """Draw prediction results"""
        height, width = image.shape[:2]
        
        if hand_info:
            for hand in hand_info:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand['landmarks'],
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                hand_label = f"{hand['handedness']} ({hand['confidence']:.2f})"
                cv2.putText(image, hand_label, (10, 30 + hand['hand_index'] * 25),
                           self.font, 0.5, (0, 255, 0), 1)
        
        if prediction is not None and confidence > self.confidence_threshold:
            pred_text = f"ASL: {prediction}"
            conf_text = f"Confidence: {confidence:.2f}"
            
            if confidence > 0.8:
                color = (0, 255, 0)
            elif confidence > 0.6:
                color = (0, 255, 255)
            elif confidence > 0.4:
                color = (0, 165, 255)
            else:
                color = (0, 100, 255)
            
            text_size = cv2.getTextSize(pred_text, self.font, self.font_scale, self.thickness)[0]
            cv2.rectangle(image, (width - text_size[0] - 30, 10), 
                         (width - 5, 90), (0, 0, 0), -1)
            
            cv2.putText(image, pred_text, (width - text_size[0] - 25, 35),
                       self.font, self.font_scale, color, self.thickness)
            cv2.putText(image, conf_text, (width - text_size[0] - 25, 65),
                       self.font, self.font_scale - 0.2, color, 1)
        
        status_y = height - 80
        
        web_info = "Web: http://localhost:5002"
        cv2.putText(image, web_info, (10, status_y), self.font, 0.4, (100, 255, 100), 1)
        
        if self.fps_history:
            current_fps = np.mean(list(self.fps_history))
            fps_text = f"FPS: {current_fps:.1f}"
            cv2.putText(image, fps_text, (10, height - 20),
                       self.font, 0.5, (255, 255, 255), 1)
        
        instructions = [
            "Controls: 'q'-Quit, 'Enter'-Save letter, 'Space'-New line, 'r'-Reset, 't'-Threshold"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(image, instruction, (10, height - 60 + i * 20),
                       self.font, 0.4, (255, 255, 255), 1)
        
        return image
    
    def detect_from_camera(self, camera_id=None, save_predictions=True):
        """Real-time detection from camera with web server"""
        if self.model is None:
            print("No model loaded. Call load_trained_model() first!")
            return
        
        if self.scaler is None or self.pca is None:
            print("Missing scaler or PCA transformer. Cannot proceed!")
            return
        
        if camera_id is None:
            camera_id = self.get_camera_id()
            if camera_id is None:
                print("No camera selected")
                return
        
        print(f"Starting REAL ASL detection with Camera {camera_id}...")
        print("Web interface available at: http://localhost:5002")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera started successfully!")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'Enter' - Save current letter to asl_predictions.txt")
        print("  'Space' - Add newline to asl_predictions.txt")
        print("  'r' - Reset prediction history")
        print("  't' - Adjust confidence threshold")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                
                landmarks_data, hand_info, _ = self.extract_hand_landmarks(frame)
                
                prediction, confidence = None, 0.0
                if landmarks_data:
                    feature_vector = self.create_feature_vector(landmarks_data)
                    raw_prediction, raw_confidence = self.predict_sign(feature_vector)
                    prediction, confidence = self.smooth_predictions(raw_prediction, raw_confidence)
                
                # Update web interface data
                self.update_web_data(prediction)
                
                frame = self.draw_predictions(frame, prediction, confidence, hand_info)
                
                frame_time = time.time() - frame_start
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.fps_history.append(fps)
                
                cv2.imshow(f'Real ASL Detection - Camera {camera_id}', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == 13:  # Enter key
                    if prediction is not None:
                        text_filename = "asl_predictions.txt"
                        try:
                            with open(text_filename, "a", encoding="utf-8") as f:
                                f.write(str(prediction))
                            print(f"Saved letter: {prediction}")
                        except Exception as e:
                            print(f"Error writing to {text_filename}: {e}")
                    else:
                        print("No prediction to save.")
                elif key == ord(' '):  # Space bar
                    text_filename = "asl_predictions.txt"
                    try:
                        with open(text_filename, "a", encoding="utf-8") as f:
                            f.write(' ')
                        print("Added newline")
                    except Exception as e:
                        print(f"Error writing to {text_filename}: {e}")
                elif key == ord('r'):
                    self.prediction_history.clear()
                    print("Prediction history reset")
                elif key == ord('t'):
                    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                    try:
                        current_idx = thresholds.index(self.confidence_threshold)
                        next_idx = (current_idx + 1) % len(thresholds)
                    except ValueError:
                        next_idx = 0
                    self.confidence_threshold = thresholds[next_idx]
                    print(f"Confidence threshold: {self.confidence_threshold}")
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print(f"\nDETECTION SESSION COMPLETE")
            print(f"Total frames: {frame_count}")
            print(f"Average FPS: {avg_fps:.1f}")

def run_flask():
    """Run Flask server in a separate thread"""
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)

def main():
    # Start Flask server in background
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Web server started at http://localhost:5002")
    time.sleep(1)
    
    detector = ASLHandSignDetector()
    
    model_dir = input("Enter model directory (default: 'trained_models'): ").strip()
    if not model_dir:
        model_dir = "trained_models"
    
    detector.model_dir = model_dir
    
    if not detector.load_trained_model():
        print("Failed to load model.")
        return
    
    detector.detect_from_camera()

if __name__ == "__main__":
    main()

####if no venv created yet, run the following commands in terminal: python3.12 -m venv venv
#####source venv/bin/activate
#####pip install opencv-python mediapipe numpy pandas scikit-learn joblib matplotlib seaborn flask glob2

####then run: python3 f3.py
####deativate 
####run f1.py to add data and train model, **make sure to run 3 to save the data before exiting f1.py
####run f2.py after f1.py to train the model
####then run f3.py to use the trained model with web interface
####the data is sent to http::localhost:5002/ and asl_scrapper in the hardware folder can be used to read the data and convert to speech