"""
LSL EEG Acquisition Module
Real-time EEG data acquisition using Lab Streaming Layer
"""

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from collections import deque
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGAcquisition:
    def __init__(self, buffer_seconds=10, fs=256):
        self.buffer_seconds = buffer_seconds
        self.fs = fs
        self.buf = deque(maxlen=buffer_seconds * fs)
        self.inlet = None
        self.running = False
        self.thread = None
        
    def connect(self, timeout=5):
        """Connect to LSL EEG stream"""
        logger.info("Searching for EEG stream...")
        streams = resolve_byprop('type', 'EEG', timeout=timeout)
        
        if not streams:
            raise RuntimeError("No LSL EEG stream found")
            
        logger.info(f"Found EEG stream: {streams[0].name()}")
        self.inlet = StreamInlet(streams[0])
        
        # Get stream info
        info = self.inlet.info()
        self.fs = int(info.nominal_srate())
        self.n_channels = info.channel_count()
        
        logger.info(f"Connected: {self.n_channels} channels @ {self.fs}Hz")
        
    def start_acquisition(self):
        """Start background acquisition thread"""
        if self.inlet is None:
            raise RuntimeError("Must connect to stream first")
            
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Acquisition started")
        
    def stop_acquisition(self):
        """Stop acquisition thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        logger.info("Acquisition stopped")
        
    def _acquisition_loop(self):
        """Main acquisition loop"""
        logger.info("Starting acquisition loop...")
        while self.running:
            try:
                sample, timestamp = self.inlet.pull_sample(timeout=0.001)
                if sample:
                    self.buf.append((timestamp, sample))
            except Exception as e:
                logger.error(f"Acquisition error: {e}")
                time.sleep(0.001)
                
    def get_latest_window(self, window_seconds=1.0):
        """Get latest EEG window"""
        if len(self.buf) == 0:
            return None, None
            
        # Convert to arrays
        timestamps = np.array([item[0] for item in self.buf])
        data = np.array([item[1] for item in self.buf])
        
        # Get latest window
        latest_time = timestamps[-1]
        start_time = latest_time - window_seconds
        
        mask = timestamps >= start_time
        if np.sum(mask) < window_seconds * self.fs * 0.8:  # At least 80% of expected samples
            return None, None
            
        window_data = data[mask].T  # Shape: (channels, samples)
        window_times = timestamps[mask]
        
        return window_data, window_times
        
    def get_epoch_around_timestamp(self, target_timestamp, pre_seconds=0.2, post_seconds=0.8):
        """Extract epoch around specific timestamp (for P300 analysis)"""
        if len(self.buf) == 0:
            return None, None
            
        timestamps = np.array([item[0] for item in self.buf])
        data = np.array([item[1] for item in self.buf])
        
        # Define epoch window
        epoch_start = target_timestamp - pre_seconds
        epoch_end = target_timestamp + post_seconds
        
        mask = (timestamps >= epoch_start) & (timestamps <= epoch_end)
        
        # Check if we have sufficient data
        expected_samples = int((pre_seconds + post_seconds) * self.fs)
        if np.sum(mask) < expected_samples * 0.7:
            return None, None
            
        epoch_data = data[mask].T  # Shape: (channels, samples)
        epoch_times = timestamps[mask]
        
        return epoch_data, epoch_times
        
    def get_buffer_status(self):
        """Get buffer status info"""
        return {
            'buffer_size': len(self.buf),
            'max_size': self.buf.maxlen,
            'duration_seconds': len(self.buf) / self.fs if self.fs > 0 else 0,
            'is_running': self.running
        }

def main():
    """Test acquisition system"""
    acq = EEGAcquisition()
    
    try:
        # Connect to stream
        acq.connect()
        
        # Start acquisition
        acq.start_acquisition()
        
        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            
            # Get latest window
            data, times = acq.get_latest_window(1.0)
            
            if data is not None:
                logger.info(f"Window {i+1}: {data.shape} samples, "
                          f"range: [{data.min():.2f}, {data.max():.2f}]")
            else:
                logger.warning(f"Window {i+1}: No data available")
                
            # Buffer status
            status = acq.get_buffer_status()
            logger.info(f"Buffer: {status['buffer_size']}/{status['max_size']} "
                       f"({status['duration_seconds']:.1f}s)")
                       
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        acq.stop_acquisition()

if __name__ == "__main__":
    main()