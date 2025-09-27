import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import signal
import pickle

class HeartRateModel:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def extract_features(self, rgb_signals, age=25, gender='M'):
        """Extract heart rate relevant features from RGB signals"""
        if len(rgb_signals) < 30:
            return np.zeros(20)  # Return zeros if not enough frames
        
        features = []
        
        # For each RGB channel
        for channel in range(3):
            channel_signal = rgb_signals[:, channel]
            
            # Detrend signal
            detrended = signal.detrend(channel_signal)
            
            # FFT features (focus on 0.5-4 Hz range for HR)
            fft = np.abs(np.fft.fft(detrended))
            freqs = np.fft.fftfreq(len(detrended), d=1/30)  # Assuming 30 fps
            
            hr_range = (freqs >= 0.5) & (freqs <= 4.0)
            if np.any(hr_range):
                dominant_freq = freqs[hr_range][np.argmax(fft[hr_range])]
                features.append(dominant_freq * 60)  # Convert to BPM
                features.append(np.max(fft[hr_range]))  # Peak magnitude
            else:
                features.extend([70.0, 0.1])  # Default values
            
            # Statistical features
            features.extend([
                np.mean(channel_signal),
                np.std(channel_signal),
                np.var(channel_signal)
            ])
        
        # Demographic features
        features.append(age)
        features.append(1 if gender == 'M' else 0)
        
        # Inter-channel features
        rgb_mean = np.mean(rgb_signals, axis=0)
        features.extend([
            rgb_mean[1] - rgb_mean[0],  # G-R
            rgb_mean[1] - rgb_mean[2],  # G-B
        ])
        
        return np.array(features)
    
    def train_sample(self, rgb_signals, hr_gt, age=25, gender='M'):
        """Train on a single sample"""
        features = self.extract_features(rgb_signals, age, gender).reshape(1, -1)
        
        if not self.is_trained:
            # Initialize with first sample
            self.scaler.fit(features)
            self.X_train = self.scaler.transform(features)
            self.y_train = np.array([hr_gt])
            self.is_trained = True
        else:
            # Add to training data
            features_scaled = self.scaler.transform(features)
            self.X_train = np.vstack([self.X_train, features_scaled])
            self.y_train = np.append(self.y_train, hr_gt)
        
        # Retrain model (RandomForest handles incremental learning well)
        self.model.fit(self.X_train, self.y_train)
        
        return self.predict_sample(rgb_signals, age, gender)
    
    def predict_sample(self, rgb_signals, age=25, gender='M'):
        """Predict heart rate for a single sample"""
        if not self.is_trained:
            return 70.0  # Default HR
        
        features = self.extract_features(rgb_signals, age, gender).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        # Clamp to reasonable HR range
        return max(40, min(200, prediction))
    
    def save_model(self, path="hr_model.pkl"):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'X_train': self.X_train if self.is_trained else None,
            'y_train': self.y_train if self.is_trained else None
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path="hr_model.pkl"):
        """Load trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        self.X_train = model_data['X_train']
        self.y_train = model_data['y_train']