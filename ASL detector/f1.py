import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime

class ASLDataCollector: 
    def __init__(self):
        """
        Collect real ASL hand sign data with proper labels
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focus on one hand
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Real ASL letters we'll collect
        self.asl_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N' , 'O', 'S', 'T', 'V', 'Y', 'HELLO', '67']
        
        # Data storage
        self.collected_data = []
        
        # Create data directory
        self.data_dir = "real_asl_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    
    def extract_landmarks(self, image):
        """Extract hand landmarks"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image.flags.writeable = False
        
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract coordinates
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return landmarks, hand_landmarks
        
        return None, None
    
    def collect_letter_data(self, letter, samples_needed=100):
        """Collect data for a specific ASL letter, appending to existing data if present"""
        if letter not in self.asl_letters:
            print(f" Invalid letter: {letter}")
            return
        
        # Load existing per-letter data if it exists
        letter_filename = os.path.join(self.data_dir, f"{letter}_samples.json")
        existing_letter_data = []
        starting_sample_id = 0
        if os.path.exists(letter_filename):
            try:
                with open(letter_filename, 'r') as f:
                    existing_letter_data = json.load(f)
                if existing_letter_data:
                    existing_ids = [s.get("sample_id", 0) for s in existing_letter_data]
                    starting_sample_id = max(existing_ids) + 1
                print(f"Found existing data for letter {letter}: {len(existing_letter_data)} samples")
                print(f"New samples will start from sample_id {starting_sample_id}")
            except Exception as e:
                print(f"Warning: could not read existing {letter_filename}: {e}")
        
        print(f"\n COLLECTING DATA FOR LETTER '{letter}'")
        print(f"Target: {samples_needed} new samples")
        print("="*40)
        print("Instructions:")
        print(f"1. Make the ASL sign for letter '{letter}'")
        print("2. Hold the sign steady")
        print("3. Press SPACE when ready to capture")
        print("4. Press 'q' to finish this letter")
        print("5. Press 'r' to restart this letter (this session only)")
        print("="*40)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        collected_count = 0
        # This will hold only the *new* samples for this run
        new_letter_data = []
        
        # Show reference image for the letter (if available)
        print(f"\n REFERENCE: Please make the ASL sign for '{letter}'")
        if letter == 'A':
            print("    Closed fist with thumb to the side")
        elif letter == 'B':
            print("    Open palm, fingers together, thumb across palm")
        elif letter == 'C':
            print("    Curved hand like holding a cup")
        elif letter == 'D':
            print("    Index finger up, other fingers touch thumb")
        elif letter == 'E':
            print("    Closed fist with fingertips touching thumb")
        elif letter == 'F':
            print("    OK sign - thumb and index finger circle")
        elif letter == 'G':
            print("    Index finger pointing sideways")
        elif letter == 'H':
            print("    Two fingers pointing sideways")
        elif letter == 'I':
            print("    Pinky finger up")
        elif letter == 'L':
            print("    Thumb and index finger make L shape")
        elif letter == 'O':
            print("    All fingertips touch thumb in circle")
        elif letter == 'V':
            print("   ️ Two fingers up in V shape")
        elif letter == 'Y':
            print("    Thumb and pinky extended")
        
        while collected_count < samples_needed:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror image
            
            # Extract landmarks
            landmarks, hand_landmarks_obj = self.extract_landmarks(frame)
            
            # Draw landmarks if detected
            if hand_landmarks_obj:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks_obj, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
            
            # Add UI text
            cv2.putText(frame, f"Letter: {letter}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Samples (this session): {collected_count}/{samples_needed}", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            if landmarks:
                cv2.putText(frame, "Hand detected - Press SPACE to capture", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Hold the sign steady!", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No hand detected", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Progress bar
            bar_width = 400
            bar_height = 20
            bar_x = 10
            bar_y = 180
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            progress_width = int((collected_count / samples_needed) * bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
            
            # Instructions
            cv2.putText(frame, "SPACE: Capture | Q: Finish | R: Restart (this letter)", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(f'ASL Data Collection - Letter {letter}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and landmarks:
                # Capture sample
                sample = {
                    'letter': letter,
                    'landmarks': landmarks,
                    'timestamp': datetime.now().isoformat(),
                    'sample_id': starting_sample_id + collected_count,
                    'user': '2PMusername'
                }
                new_letter_data.append(sample)
                collected_count += 1
                print(f" Captured sample {collected_count}/{samples_needed} for letter {letter} (global sample_id {sample['sample_id']})")
                
            elif key == ord('q'):
                break
            elif key == ord('r'):
                # Restart collection for this letter (in this session only)
                new_letter_data = []
                collected_count = 0
                print(f" Restarting collection for letter {letter} (this session)")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if new_letter_data:
            # Merge existing + new data and save
            merged_letter_data = existing_letter_data + new_letter_data
            try:
                with open(letter_filename, 'w') as f:
                    json.dump(merged_letter_data, f, indent=2)
                print(f" Saved {len(new_letter_data)} new samples for letter {letter}")
                print(f" Total samples for {letter}: {len(merged_letter_data)}")
                print(f" File: {letter_filename}")
            except Exception as e:
                print(f"Error saving {letter_filename}: {e}")
            
            # Add new samples to main collection for this run
            self.collected_data.extend(new_letter_data)
        
        return new_letter_data
    
    def collect_all_letters(self, samples_per_letter=50):
        """Collect data for all ASL letters (appends to existing data)"""
        print(f"\n COLLECTING DATA FOR ALL {len(self.asl_letters)} LETTERS")
        print(f" Target: {samples_per_letter} new samples per letter")
        print(f" Total target (this run): {len(self.asl_letters) * samples_per_letter} samples")
        
        for i, letter in enumerate(self.asl_letters, 1):
            print(f"\n--- Letter {i}/{len(self.asl_letters)}: {letter} ---")
            
            # Ask if user wants to collect this letter
            while True:
                choice = input(f"Collect data for letter '{letter}'? (y/n/skip): ").strip().lower()
                if choice in ['y', 'yes']:
                    break
                elif choice in ['n', 'no']:
                    print(" Exiting collection")
                    return
                elif choice == 'skip':
                    print(f" Skipping letter {letter}")
                    break
                else:
                    print(" Please enter 'y', 'n', or 'skip'")
            
            if choice == 'skip':
                continue
            
            # Collect data for this letter
            letter_data = self.collect_letter_data(letter, samples_per_letter)
            
            if letter_data:
                print(f"Completed letter {letter}: {len(letter_data)} new samples")
            
            # Ask if user wants to continue
            if i < len(self.asl_letters):
                while True:
                    continue_choice = input("Continue to next letter? (y/n): ").strip().lower()
                    if continue_choice in ['y', 'yes']:
                        break
                    elif continue_choice in ['n', 'no']:
                        print(" Stopping collection")
                        return
                    else:
                        print(" Please enter 'y' or 'n'")
        
        print(f"\n COLLECTION COMPLETE (this run)!")
        print(f" New samples collected this run: {len(self.collected_data)}")
        
        # Create/extend final dataset
        self.create_training_dataset()
    
    def create_training_dataset(self):
        """Create or update final training dataset CSV/JSON by appending to existing data"""
        # If nothing collected this run, still try to build from disk
        if not self.collected_data:
            print(" No new data collected in this run, building dataset from files on disk...")
            # Load all per-letter JSON files
            all_samples = []
            for letter in self.asl_letters:
                letter_filename = os.path.join(self.data_dir, f"{letter}_samples.json")
                if os.path.exists(letter_filename):
                    try:
                        with open(letter_filename, 'r') as f:
                            letter_samples = json.load(f)
                        all_samples.extend(letter_samples)
                    except Exception as e:
                        print(f"Warning: could not read {letter_filename}: {e}")
            if not all_samples:
                print(" No data found on disk either.")
                return
            base_data = all_samples
        else:
            base_data = self.collected_data
        
        print("\n Creating/updating training dataset...")
        
        # Convert to DataFrame format (new data for this operation)
        rows = []
        for sample in base_data:
            row = {
                'letter': sample['letter'],
                'timestamp': sample['timestamp'],
                'sample_id': sample['sample_id'],
                'user': sample['user']
            }
            
            # Add landmark coordinates
            landmarks = sample['landmarks']
            for i, coord in enumerate(landmarks):
                row[f'landmark_{i}'] = coord
            
            rows.append(row)
        
        df_new = pd.DataFrame(rows)
        
        csv_filename = os.path.join(self.data_dir, 'real_asl_dataset.csv')
        json_filename = os.path.join(self.data_dir, 'real_asl_dataset.json')
        
        # If existing dataset files exist, load and append to them
        if os.path.exists(csv_filename):
            try:
                df_existing = pd.read_csv(csv_filename)
                # Align columns (in case of schema drift)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False)
                # Drop exact duplicates across all columns
                df_combined = df_combined.drop_duplicates()
                df_final = df_combined
                print(f" Existing CSV found: {csv_filename}")
                print(f" Existing rows: {len(df_existing)}, new rows: {len(df_new)}, merged rows: {len(df_final)}")
            except Exception as e:
                print(f"Warning: could not read existing CSV, recreating from new data only. Error: {e}")
                df_final = df_new
        else:
            df_final = df_new
        
        # Save CSV
        df_final.to_csv(csv_filename, index=False)
        
        # Handle JSON: merge with existing if exists
        existing_json_data = []
        if os.path.exists(json_filename):
            try:
                with open(json_filename, 'r') as f:
                    existing_json_data = json.load(f)
            except Exception as e:
                print(f"Warning: could not read existing JSON, recreating from new data only. Error: {e}")
        
        # Convert df_final back to list-of-dicts JSON format compatible with original structure
        # We assume each row corresponds to a sample with flattened landmarks.
        # For JSON, we prefer keeping the original structure with 'landmarks' list.
        # So we reconstruct from column names.
        final_samples_json = []
        for _, row in df_final.iterrows():
            sample_dict = {
                'letter': row['letter'],
                'timestamp': row['timestamp'],
                'sample_id': int(row['sample_id']) if not pd.isna(row['sample_id']) else None,
                'user': row['user']
            }
            # Collect landmark_* columns in order
            landmarks = []
            i = 0
            while f'landmark_{i}' in row.index:
                landmarks.append(float(row[f'landmark_{i}']))
                i += 1
            sample_dict['landmarks'] = landmarks
            final_samples_json.append(sample_dict)
        
        # Merge existing JSON with new, dropping exact duplicates
        # (We use tuple of sorted items as a simple dedup key)
        merged_json = existing_json_data + final_samples_json
        unique_seen = set()
        deduped_json = []
        for s in merged_json:
            key = (
                s.get('letter'),
                s.get('timestamp'),
                s.get('sample_id'),
                tuple(s.get('landmarks', [])),
                s.get('user'),
            )
            if key not in unique_seen:
                unique_seen.add(key)
                deduped_json.append(s)
        
        with open(json_filename, 'w') as f:
            json.dump(deduped_json, f, indent=2)
        
        print(f" Training dataset updated!")
        print(f"    CSV: {csv_filename}")
        print(f"   JSON: {json_filename}")
        print(f"    Total samples: {len(df_final)}")
        
        # Show distribution
        print(f"\n Letter distribution:")
        letter_counts = df_final['letter'].value_counts().sort_index()
        for letter, count in letter_counts.items():
            print(f"   {letter}: {count} samples")
        
        return csv_filename

def main():
    """Main function for ASL data collection"""
    print(" REAL ASL DATA COLLECTOR")
    print("="*50)
    
    collector = ASLDataCollector()
    
    print("\nOptions:")
    print("1. Collect data for single letter")
    print("2. Collect data for all letters")
    print("3. Create/update dataset from existing data")
    print("4. Exit")
    print("5. Add new option")
    
    while True:
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            letter = input("Enter letter to collect (A-Z): ").strip().upper()
            if letter in collector.asl_letters:
                samples = int(input(f"How many new samples for {letter}? (default 50): ") or 50)
                collector.collect_letter_data(letter, samples)
            else:
                print(f" Letter {letter} not in collection list: {collector.asl_letters}")
                
        elif choice == '2':
            samples = int(input("New samples per letter? (default 30): ") or 30)
            collector.collect_all_letters(samples)
            break
            
        elif choice == '3':
            collector.create_training_dataset()
            break
            
        elif choice == '4':
            print(" Goodbye!")
            break
        elif choice == '5':
            collector.asl_letters.append(input("Enter new letter/sign to add: ").strip().upper())
        
        else:
            print(" Please enter 1, 2, 3, 4, or 5")

if __name__ == "__main__":
    main()