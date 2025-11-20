"""
P300 Online Processing Module
Real-time P300 detection and fatigue index computation
"""

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from collections import deque
import time
import logging

logger = logging.getLogger(__name__)

class P300OnlineProcessor:
    def __init__(self, fs=256, epoch_window=(-0.2, 0.8), baseline_window=(-0.2, 0.0)):
        self.fs = fs
        self.epoch_window = epoch_window  # seconds
        self.baseline_window = baseline_window  # seconds
        
        # Convert to samples
        self.epoch_samples = (int(epoch_window[0] * fs), int(epoch_window[1] * fs))
        self.baseline_samples = (int(baseline_window[0] * fs), int(baseline_window[1] * fs))
        
        # P300 parameters
        self.p300_window = (0.25, 0.65)  # P300 peak window in seconds
        self.p300_samples = (int(self.p300_window[0] * fs), int(self.p300_window[1] * fs))
        
        # Running averages
        self.running_amplitude = None
        self.running_latency = None
        self.alpha = 0.05  # Smoothing factor
        
        # Fatigue tracking
        self.amplitude_history = deque(maxlen=100)  # Last 100 trials
        self.fatigue_index = 0.0
        self.baseline_amplitude = None  # First 30 trials average
        
        # ERP waveform storage
        self.erp_waveforms = deque(maxlen=50)  # Store last 50 ERPs
        self.current_erp = None
        
        # EEG buffer for epoching
        self.eeg_buffer = deque(maxlen=int(fs * 5))  # 5 second buffer
        self.marker_inlet = None
        
        # Trial tracking
        self.trial_count = 0
        self.oddball_erps = deque(maxlen=30)  # Oddball ERPs
        self.standard_erps = deque(maxlen=30)  # Standard ERPs
        
        # Channel selection (prefer Cz for P300)
        self.target_channels = ['Cz', 'Pz', 'CPz', 'C1', 'C2']
        self.channel_index = 0  # Default to first channel
        
    def connect_marker_stream(self, timeout=5):
        """Connect to LSL marker stream"""
        logger.info("Searching for marker stream...")
        streams = resolve_byprop('type', 'Markers', timeout=timeout)
        
        if not streams:
            logger.warning("No marker stream found - P300 processing disabled")
            return False
            
        self.marker_inlet = StreamInlet(streams[0])
        logger.info(f"Connected to marker stream: {streams[0].name()}")
        return True
        
    def set_channel_info(self, channel_names):
        """Set channel information for P300 analysis"""
        # Find best channel for P300 (prefer Cz)
        for target in self.target_channels:
            if target in channel_names:
                self.channel_index = channel_names.index(target)
                logger.info(f"Using channel {target} (index {self.channel_index}) for P300")
                return
                
        # Default to first channel if no preferred channels found
        self.channel_index = 0
        logger.warning(f"No preferred P300 channels found, using {channel_names[0]}")
        
    def update_eeg_buffer(self, eeg_data, timestamps):
        """Update EEG buffer with new data"""
        for i, timestamp in enumerate(timestamps):
            sample = eeg_data[:, i] if eeg_data.ndim > 1 else [eeg_data[i]]
            self.eeg_buffer.append((timestamp, sample))
            
    def process_markers(self):
        """Process incoming markers and extract P300 epochs"""
        if self.marker_inlet is None:
            return None
            
        # Check for new markers
        marker, timestamp = self.marker_inlet.pull_sample(timeout=0.0)
        
        if marker is None:
            return None
            
        # Process marker
        marker_type = marker[0] if isinstance(marker, list) else marker
        logger.debug(f"Received marker: {marker_type} at {timestamp}")
        
        # Extract epoch around marker
        epoch_data = self._extract_epoch(timestamp)
        
        if epoch_data is not None:
            # Process P300
            p300_result = self._process_p300_epoch(epoch_data, marker_type)
            return p300_result
            
        return None
        
    def _extract_epoch(self, marker_timestamp):
        """Extract EEG epoch around marker timestamp"""
        if len(self.eeg_buffer) == 0:
            return None
            
        # Convert buffer to arrays
        timestamps = np.array([item[0] for item in self.eeg_buffer])
        data = np.array([item[1] for item in self.eeg_buffer])
        
        # Find samples within epoch window
        epoch_start = marker_timestamp + self.epoch_window[0]
        epoch_end = marker_timestamp + self.epoch_window[1]
        
        mask = (timestamps >= epoch_start) & (timestamps <= epoch_end)
        
        if np.sum(mask) < 0.8 * (self.epoch_samples[1] - self.epoch_samples[0]):
            logger.warning("Insufficient data for epoch extraction")
            return None
            
        epoch_data = data[mask]
        epoch_times = timestamps[mask]
        
        return {
            'data': epoch_data,
            'timestamps': epoch_times,
            'marker_time': marker_timestamp
        }
        
    def _process_p300_epoch(self, epoch_data, marker_type):
        """Process P300 epoch and compute amplitude/latency"""
        try:
            data = epoch_data['data']
            timestamps = epoch_data['timestamps']
            marker_time = epoch_data['marker_time']
            
            # Get target channel data
            if data.ndim > 1 and data.shape[1] > self.channel_index:
                channel_data = data[:, self.channel_index]
            else:
                channel_data = data.flatten()
                
            # Baseline correction
            baseline_start = marker_time + self.baseline_window[0]
            baseline_end = marker_time + self.baseline_window[1]
            baseline_mask = (timestamps >= baseline_start) & (timestamps <= baseline_end)
            
            if np.sum(baseline_mask) > 0:
                baseline_mean = np.mean(channel_data[baseline_mask])
                channel_data = channel_data - baseline_mean
                
            # Store ERP waveform (interpolate to standard time grid)
            epoch_times = timestamps - marker_time  # Relative to marker
            standard_times = np.linspace(self.epoch_window[0], self.epoch_window[1], 
                                       int((self.epoch_window[1] - self.epoch_window[0]) * self.fs))
            
            # Interpolate to standard grid
            erp_waveform = np.interp(standard_times, epoch_times, channel_data)
            self.current_erp = erp_waveform
            self.erp_waveforms.append(erp_waveform.copy())
            
            # Store by trial type
            if marker_type in ['TARGET', 'oddball']:
                self.oddball_erps.append(erp_waveform.copy())
            elif marker_type in ['NONTARGET', 'standard']:
                self.standard_erps.append(erp_waveform.copy())
                
            # Find P300 peak in time window
            p300_start = marker_time + self.p300_window[0]
            p300_end = marker_time + self.p300_window[1]
            p300_mask = (timestamps >= p300_start) & (timestamps <= p300_end)
            
            if np.sum(p300_mask) == 0:
                return None
                
            p300_data = channel_data[p300_mask]
            p300_times = timestamps[p300_mask]
            
            # Find peak (maximum positive deflection for P300)
            peak_idx = np.argmax(p300_data)
            amplitude = p300_data[peak_idx]
            latency = p300_times[peak_idx] - marker_time
            
            # Update trial count
            self.trial_count += 1
            
            # Update running averages
            if self.running_amplitude is None:
                self.running_amplitude = amplitude
                self.running_latency = latency
            else:
                self.running_amplitude = (1 - self.alpha) * self.running_amplitude + self.alpha * amplitude
                self.running_latency = (1 - self.alpha) * self.running_latency + self.alpha * latency
                
            # Update fatigue tracking
            self.amplitude_history.append(amplitude)
            self._update_fatigue_index()
            
            result = {
                'amplitude': amplitude,
                'latency': latency * 1000,  # Convert to ms
                'running_amplitude': self.running_amplitude,
                'running_latency': self.running_latency * 1000,
                'fatigue_index': self.fatigue_index,
                'marker_type': marker_type,
                'timestamp': marker_time,
                'erp_waveform': erp_waveform,
                'trial_number': self.trial_count
            }
            
            logger.debug(f"P300: amp={amplitude:.2f}, lat={latency*1000:.1f}ms, "
                        f"fatigue={self.fatigue_index:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"P300 processing error: {e}")
            return None
            
    def _update_fatigue_index(self):
        """Update fatigue index based on amplitude decline (as per protocol)"""
        if len(self.amplitude_history) < 10:
            self.fatigue_index = 0.0
            return
            
        # Get current amplitude
        current_amplitude = self.amplitude_history[-1]
        
        # Establish baseline from first 30 trials
        if self.baseline_amplitude is None and len(self.amplitude_history) >= 30:
            self.baseline_amplitude = np.mean(list(self.amplitude_history)[:30])
            
        if self.baseline_amplitude is not None and self.baseline_amplitude > 0:
            # Fatigue index as per protocol: fatigue = 1 - (amplitude / baseline)
            self.fatigue_index = max(0, 1 - (current_amplitude / self.baseline_amplitude))
            self.fatigue_index = min(1, self.fatigue_index)  # Clip to [0, 1]
        else:
            # Fallback: use trend-based approach
            amplitudes = np.array(list(self.amplitude_history))
            x = np.arange(len(amplitudes))
            slope = np.polyfit(x, amplitudes, 1)[0]
            
            baseline_amplitude = np.mean(amplitudes[:min(30, len(amplitudes))])
            if baseline_amplitude > 0:
                fatigue_rate = -slope / baseline_amplitude
                self.fatigue_index = np.clip(fatigue_rate, 0, 1)
            else:
                self.fatigue_index = 0.0
            
    def get_average_erp(self, trial_type='all', n_trials=30):
        """Get average ERP waveform"""
        if trial_type == 'oddball' and len(self.oddball_erps) > 0:
            erps = list(self.oddball_erps)[-n_trials:]
        elif trial_type == 'standard' and len(self.standard_erps) > 0:
            erps = list(self.standard_erps)[-n_trials:]
        elif trial_type == 'all' and len(self.erp_waveforms) > 0:
            erps = list(self.erp_waveforms)[-n_trials:]
        else:
            return None
            
        if len(erps) == 0:
            return None
            
        return np.mean(erps, axis=0)
        
    def get_erp_times(self):
        """Get time vector for ERP waveforms"""
        return np.linspace(self.epoch_window[0], self.epoch_window[1], 
                          int((self.epoch_window[1] - self.epoch_window[0]) * self.fs))
    
    def get_status(self):
        """Get current P300 processor status"""
        return {
            'connected': self.marker_inlet is not None,
            'running_amplitude': self.running_amplitude,
            'running_latency': self.running_latency * 1000 if self.running_latency else None,
            'fatigue_index': self.fatigue_index,
            'trials_processed': len(self.amplitude_history),
            'channel_index': self.channel_index,
            'trial_count': self.trial_count,
            'oddball_trials': len(self.oddball_erps),
            'standard_trials': len(self.standard_erps),
            'baseline_amplitude': self.baseline_amplitude
        }

def main():
    """Test P300 processor"""
    processor = P300OnlineProcessor()
    
    # Try to connect to marker stream
    if not processor.connect_marker_stream():
        logger.error("No marker stream available for testing")
        return
        
    # Simulate some processing
    logger.info("P300 processor ready. Send some markers to test...")
    
    try:
        for i in range(100):  # Process for 10 seconds
            result = processor.process_markers()
            
            if result:
                logger.info(f"P300 detected: {result}")
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Stopped by user")
        
    # Print final status
    status = processor.get_status()
    logger.info(f"Final status: {status}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()