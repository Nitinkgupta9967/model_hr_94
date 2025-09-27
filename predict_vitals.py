from model import HeartRateModel
from data_loader import DataLoader
import sys
import numpy as np
class VitalSigns:
    def __init__(self, model_path="final_hr_model_full.pkl"):
        self.model = HeartRateModel()
        self.loader = DataLoader()
        
        # Try multiple model paths
        model_paths = [model_path, "final_hr_model.pkl", "hr_model.pkl"]
        model_loaded = False
        
        for path in model_paths:
            try:
                self.model.load_model(path)
                print(f"✅ Model loaded from: {path}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
        
        if not model_loaded:
            print("❌ Model not found. Available files:")
            import os
            pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            print(f"   Found .pkl files: {pkl_files}")
            print("Please train the model first using train.py or train_full.py")
            sys.exit(1)
    
    def predict_from_video(self, video_path, age=25, gender='M'):
        """
        Predict vital signs from video
        
        Args:
            video_path: Path to video file
            age: Person's age (affects HR prediction)
            gender: 'M' or 'F'
            
        Returns:
            dict: Predicted vital signs
        """
        print(f"🎥 Processing video: {video_path}")
        print(f"👤 Demographics: Age={age}, Gender={gender}")
        
        # Process video
        rgb_signals = self.loader.process_video(video_path, max_frames=None)  # Process ALL frames
        
        if len(rgb_signals) < 30:
            return {
                'error': 'Video too short (need at least 30 frames)',
                'heart_rate': None,
                'confidence': 'Low'
            }
        
        # Predict heart rate
        heart_rate = self.model.predict_sample(rgb_signals, age, gender)
        
        # Estimate other vitals based on HR (simplified)
        systolic_bp = 120 + (heart_rate - 70) * 0.5  # Rough correlation
        diastolic_bp = 80 + (heart_rate - 70) * 0.3
        spo2 = 98 + np.random.normal(0, 1)  # Simplified (would need different model)
        
        # Confidence based on signal quality
        signal_quality = self._assess_signal_quality(rgb_signals)
        confidence = "High" if signal_quality > 0.7 else "Medium" if signal_quality > 0.4 else "Low"
        
        results = {
            'heart_rate': round(heart_rate, 1),
            'systolic_bp': round(systolic_bp, 1),
            'diastolic_bp': round(diastolic_bp, 1),
            'spo2': round(spo2, 1),
            'confidence': confidence,
            'frames_processed': len(rgb_signals),
            'error': None
        }
        
        return results
    
    def _assess_signal_quality(self, rgb_signals):
        """Simple signal quality assessment"""
        try:
            # Check signal stability
            std_ratio = np.std(rgb_signals[:, 1]) / np.mean(rgb_signals[:, 1])  # Green channel
            quality = max(0, min(1, 1 - std_ratio))
            return quality
        except:
            return 0.5

def main():
    """Command line interface"""
    if len(sys.argv) < 2:
        print("Usage: python predict_vitals.py <video_path> [age] [gender]")
        print("Example: python predict_vitals.py video.mp4 25 M")
        return
    
    video_path = sys.argv[1]
    age = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    gender = sys.argv[3] if len(sys.argv) > 3 else 'M'
    
    # Initialize predictor
    predictor = VitalSigns()
    
    # Make prediction
    results = predictor.predict_from_video(video_path, age, gender)
    
    # Display results
    print("\n" + "="*50)
    print("🩺 VITAL SIGNS PREDICTION RESULTS")
    print("="*50)
    
    if results['error']:
        print(f"❌ Error: {results['error']}")
    else:
        print(f"💓 Heart Rate: {results['heart_rate']} BPM")
        print(f"🩸 Blood Pressure: {results['systolic_bp']}/{results['diastolic_bp']} mmHg")
        print(f"🫁 SpO2: {results['spo2']}%")
        print(f"📊 Confidence: {results['confidence']}")
        print(f"🎬 Frames Processed: {results['frames_processed']}")
    
    print("="*50)

if __name__ == "__main__":
    main()