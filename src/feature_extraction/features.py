"""
Cognitive Feature Extraction Module (CFEM)
Advanced EEG feature extraction for cognitive state classification
"""

import numpy as np
from scipy.signal import welch, periodogram
from scipy.stats import entropy
import pywt
import logging

logger = logging.getLogger(__name__)

class CFEMExtractor:
    def __init__(self, fs=256, window_length=1.0):
        self.fs = fs
        self.window_length = window_length
        self.n_samples = int(fs * window_length)
        
        # Frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 57)
        }
        
        # Channel groups for spatial features
        self.frontal_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz']
        self.parietal_channels = ['P3', 'P4', 'P7', 'P8', 'Pz']
        self.central_channels = ['C3', 'C4', 'Cz']
        
    def extract_features(self, eeg_data, channel_names=None):
        """
        Extract comprehensive cognitive features from EEG window
        
        Args:
            eeg_data: numpy array (channels, samples)
            channel_names: list of channel names (optional)
            
        Returns:
            dict: Feature dictionary
        """
        if eeg_data.shape[1] < self.n_samples:
            logger.warning(f"Window too short: {eeg_data.shape[1]} < {self.n_samples}")
            return None
            
        features = {}
        
        try:
            # 1. Spectral features
            spectral_features = self._extract_spectral_features(eeg_data)
            features.update(spectral_features)
            
            # 2. Statistical features  
            statistical_features = self._extract_statistical_features(eeg_data)
            features.update(statistical_features)
            
            # 3. Wavelet features
            wavelet_features = self._extract_wavelet_features(eeg_data)
            features.update(wavelet_features)
            
            # 4. Spatial features
            if channel_names:
                spatial_features = self._extract_spatial_features(eeg_data, channel_names)
                features.update(spatial_features)
                
            # 5. Connectivity features
            connectivity_features = self._extract_connectivity_features(eeg_data)
            features.update(connectivity_features)
            
            logger.debug(f"Extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def _extract_spectral_features(self, eeg_data):
        """Extract frequency domain features"""
        features = {}
        
        # Compute PSD using Welch method
        freqs, psd = welch(eeg_data, fs=self.fs, nperseg=min(256, eeg_data.shape[1]), 
                          noverlap=None, axis=1)
        
        # Band power features
        for band_name, (fmin, fmax) in self.bands.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            
            if np.sum(band_mask) == 0:
                continue
                
            # Power in each channel
            band_powers = np.mean(psd[:, band_mask], axis=1)
            
            features[f'{band_name}_power_mean'] = np.mean(band_powers)
            features[f'{band_name}_power_std'] = np.std(band_powers)
            features[f'{band_name}_power_max'] = np.max(band_powers)
            
        # Band ratios (cognitive markers)
        theta_power = features.get('theta_power_mean', 1e-10)
        alpha_power = features.get('alpha_power_mean', 1e-10) 
        beta_power = features.get('beta_power_mean', 1e-10)
        
        features['theta_beta_ratio'] = theta_power / beta_power
        features['alpha_beta_ratio'] = alpha_power / beta_power
        features['theta_alpha_ratio'] = theta_power / alpha_power
        
        # Spectral entropy
        psd_norm = psd / (np.sum(psd, axis=1, keepdims=True) + 1e-12)
        spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)
        features['spectral_entropy_mean'] = np.mean(spectral_entropy)
        features['spectral_entropy_std'] = np.std(spectral_entropy)
        
        # Peak frequency
        peak_freqs = freqs[np.argmax(psd, axis=1)]
        features['peak_freq_mean'] = np.mean(peak_freqs)
        features['peak_freq_std'] = np.std(peak_freqs)
        
        return features
    
    def _extract_statistical_features(self, eeg_data):
        """Extract time domain statistical features"""
        features = {}
        
        # Basic statistics per channel
        means = np.mean(eeg_data, axis=1)
        stds = np.std(eeg_data, axis=1)
        vars = np.var(eeg_data, axis=1)
        skews = self._skewness(eeg_data, axis=1)
        kurts = self._kurtosis(eeg_data, axis=1)
        
        # Aggregate across channels
        features['signal_mean'] = np.mean(means)
        features['signal_std'] = np.mean(stds)
        features['signal_var'] = np.mean(vars)
        features['signal_skew'] = np.mean(skews)
        features['signal_kurt'] = np.mean(kurts)
        
        # Channel variability
        features['channel_mean_std'] = np.std(means)
        features['channel_std_std'] = np.std(stds)
        
        # Zero crossing rate
        zcr = np.mean([self._zero_crossing_rate(ch) for ch in eeg_data])
        features['zero_crossing_rate'] = zcr
        
        # RMS energy
        rms = np.sqrt(np.mean(eeg_data**2, axis=1))
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    def _extract_wavelet_features(self, eeg_data):
        """Extract wavelet domain features"""
        features = {}
        
        # Use first channel or average for wavelet analysis
        signal = np.mean(eeg_data, axis=0)
        
        try:
            # Discrete wavelet transform
            coeffs = pywt.wavedec(signal, 'db4', level=5)
            
            # Energy in each level
            for i, coeff in enumerate(coeffs):
                energy = np.sum(coeff**2)
                features[f'wavelet_energy_level_{i}'] = energy
                
            # Relative energies
            total_energy = sum(np.sum(c**2) for c in coeffs)
            for i, coeff in enumerate(coeffs):
                rel_energy = np.sum(coeff**2) / (total_energy + 1e-12)
                features[f'wavelet_rel_energy_level_{i}'] = rel_energy
                
        except Exception as e:
            logger.warning(f"Wavelet extraction failed: {e}")
            # Fill with zeros if wavelet fails
            for i in range(6):
                features[f'wavelet_energy_level_{i}'] = 0.0
                features[f'wavelet_rel_energy_level_{i}'] = 0.0
                
        return features
    
    def _extract_spatial_features(self, eeg_data, channel_names):
        """Extract spatial/topographical features"""
        features = {}
        
        # Create channel mapping
        ch_map = {name: i for i, name in enumerate(channel_names)}
        
        # Frontal vs Parietal activity
        frontal_indices = [ch_map[ch] for ch in self.frontal_channels if ch in ch_map]
        parietal_indices = [ch_map[ch] for ch in self.parietal_channels if ch in ch_map]
        
        if frontal_indices and parietal_indices:
            frontal_power = np.mean(np.var(eeg_data[frontal_indices], axis=1))
            parietal_power = np.mean(np.var(eeg_data[parietal_indices], axis=1))
            
            features['frontal_power'] = frontal_power
            features['parietal_power'] = parietal_power
            features['frontal_parietal_ratio'] = frontal_power / (parietal_power + 1e-12)
            
        # Left vs Right asymmetry (if available)
        left_channels = [ch for ch in channel_names if ch.endswith('3') or ch.endswith('7')]
        right_channels = [ch for ch in channel_names if ch.endswith('4') or ch.endswith('8')]
        
        if left_channels and right_channels:
            left_indices = [ch_map[ch] for ch in left_channels if ch in ch_map]
            right_indices = [ch_map[ch] for ch in right_channels if ch in ch_map]
            
            if left_indices and right_indices:
                left_power = np.mean(np.var(eeg_data[left_indices], axis=1))
                right_power = np.mean(np.var(eeg_data[right_indices], axis=1))
                
                features['left_power'] = left_power
                features['right_power'] = right_power
                features['asymmetry_index'] = (right_power - left_power) / (right_power + left_power + 1e-12)
                
        return features
    
    def _extract_connectivity_features(self, eeg_data):
        """Extract simple connectivity features"""
        features = {}
        
        # Cross-correlation between channels
        n_channels = eeg_data.shape[0]
        
        if n_channels > 1:
            # Compute correlation matrix
            corr_matrix = np.corrcoef(eeg_data)
            
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix[mask]
            
            features['connectivity_mean'] = np.mean(correlations)
            features['connectivity_std'] = np.std(correlations)
            features['connectivity_max'] = np.max(correlations)
            features['connectivity_min'] = np.min(correlations)
            
        return features
    
    def _skewness(self, data, axis=1):
        """Compute skewness"""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-12))**3, axis=axis)
    
    def _kurtosis(self, data, axis=1):
        """Compute kurtosis"""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-12))**4, axis=axis) - 3
    
    def _zero_crossing_rate(self, signal):
        """Compute zero crossing rate"""
        return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)

def main():
    """Test feature extraction"""
    # Generate synthetic EEG data
    fs = 256
    duration = 2.0
    n_channels = 8
    n_samples = int(fs * duration)
    
    # Create synthetic data with different frequency components
    t = np.linspace(0, duration, n_samples)
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Mix of frequencies
        eeg_data[ch] = (0.5 * np.sin(2*np.pi*10*t) +  # Alpha
                       0.3 * np.sin(2*np.pi*6*t) +   # Theta  
                       0.2 * np.sin(2*np.pi*20*t) +  # Beta
                       0.1 * np.random.randn(n_samples))  # Noise
    
    # Extract features
    extractor = CFEMExtractor(fs=fs)
    features = extractor.extract_features(eeg_data)
    
    if features:
        print(f"Extracted {len(features)} features:")
        for name, value in sorted(features.items()):
            print(f"  {name}: {value:.6f}")
    else:
        print("Feature extraction failed")

if __name__ == "__main__":
    main()