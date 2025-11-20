"""
ML Inference Server
Real-time cognitive state prediction with low latency
"""

import joblib
import time
import numpy as np
from pylsl import StreamOutlet, StreamInfo
import logging
import threading
from pathlib import Path
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from feature_extraction.features import CFEMExtractor
from acquisition.lsl_acquire import EEGAcquisition
from p300.p300_online import P300OnlineProcessor

logger = logging.getLogger(__name__)

class NAIInferenceServer:
    def __init__(self, model_path='models/nai_voting_model.pkl', fs=256):
        self.model_path = model_path
        self.fs = fs
        self.model = None
        self.feature_extractor = CFEMExtractor(fs=fs)
        self.p300_processor = P300OnlineProcessor(fs=fs)
        
        # LSL outlets for predictions
        self.state_outlet = None
        self.p300_outlet = None
        
        # EEG acquisition
        self.eeg_acq = EEGAcquisition(fs=fs)
        
        # Processing control
        self.running = False
        self.inference_thread = None
        
        # Performance tracking
        self.inference_times = []
        self.prediction_count = 0
        
        # State mapping
        self.state_names = ['Relaxed', 'Focused', 'Distracted', 'Overload']
        
    def load_model(self):
        """Load trained ML model"""
        try:
            model_file = Path(self.model_path)
            if not model_file.exists():
                # Try relative path from project root
                model_file = Path(__file__).parent.parent.parent / self.model_path
                
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            self.model = joblib.load(model_file)
            logger.info(f"Model loaded successfully from {model_file}")
            
            # Get model info
            if hasattr(self.model, 'classes_'):
                logger.info(f"Model classes: {self.model.classes_}")
            else:
                logger.warning("Model classes not available")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
            
    def setup_lsl_outlets(self):
        """Setup LSL outlets for streaming predictions"""
        try:
            # State prediction outlet
            state_info = StreamInfo('NAI_State', 'Predictions', 3, 0, 'string', 'nai_state_001')
            self.state_outlet = StreamOutlet(state_info)
            
            # P300 outlet
            p300_info = StreamInfo('NAI_P300', 'ERP', 4, 0, 'float32', 'nai_p300_001')
            self.p300_outlet = StreamOutlet(p300_info)
            
            logger.info("LSL outlets created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create LSL outlets: {e}")
            return False
            
    def start_server(self):
        """Start the inference server"""
        if not self.load_model():
            return False
            
        if not self.setup_lsl_outlets():
            return False
            
        try:
            # Connect to EEG stream
            self.eeg_acq.connect()
            self.eeg_acq.start_acquisition()
            
            # Setup P300 processor
            self.p300_processor.connect_marker_stream()
            
            # Start inference loop
            self.running = True
            self.inference_thread = threading.Thread(target=self._inference_loop)
            self.inference_thread.daemon = True
            self.inference_thread.start()
            
            logger.info("NAI Inference Server started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
            
    def stop_server(self):
        """Stop the inference server"""
        self.running = False
        
        if self.inference_thread:
            self.inference_thread.join()
            
        if self.eeg_acq:
            self.eeg_acq.stop_acquisition()
            
        logger.info("NAI Inference Server stopped")
        
    def _inference_loop(self):
        """Main inference processing loop"""
        logger.info("Starting inference loop...")
        
        step_interval = 0.25  # 250ms steps
        last_step_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                
                # Process at regular intervals
                if current_time - last_step_time >= step_interval:
                    self._process_inference_step()
                    last_step_time = current_time
                    
                # Process P300 markers (continuous)
                self._process_p300_step()
                
                time.sleep(0.001)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
                time.sleep(0.1)
                
    def _process_inference_step(self):
        """Process one inference step"""
        # Get latest EEG window
        eeg_data, timestamps = self.eeg_acq.get_latest_window(1.0)
        
        if eeg_data is None:
            return
            
        # Update P300 processor with EEG data
        self.p300_processor.update_eeg_buffer(eeg_data, timestamps)
        
        # Extract features
        t0 = time.perf_counter()
        features = self.feature_extractor.extract_features(eeg_data)
        
        if features is None:
            return
            
        # Prepare feature vector
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Make prediction
        try:
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            confidence = np.max(probabilities)
            
            # Get state name
            if hasattr(self.model, 'classes_'):
                state_name = self.model.classes_[prediction]
            else:
                state_name = self.state_names[prediction] if prediction < len(self.state_names) else f"State_{prediction}"
                
            inference_time = (time.perf_counter() - t0) * 1000  # ms
            self.inference_times.append(inference_time)
            self.prediction_count += 1
            
            # Send prediction via LSL
            self.state_outlet.push_sample([
                str(state_name),
                f"{confidence:.3f}",
                f"{inference_time:.1f}"
            ])
            
            # Log prediction
            if self.prediction_count % 10 == 0:  # Log every 10th prediction
                avg_latency = np.mean(self.inference_times[-10:])
                logger.info(f"Prediction #{self.prediction_count}: {state_name} "
                          f"(conf={confidence:.3f}, lat={avg_latency:.1f}ms)")
                          
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            
    def _process_p300_step(self):
        """Process P300 detection step"""
        p300_result = self.p300_processor.process_markers()
        
        if p300_result:
            # Send P300 data via LSL
            self.p300_outlet.push_sample([
                p300_result['amplitude'],
                p300_result['latency'],
                p300_result['running_amplitude'],
                p300_result['fatigue_index']
            ])
            
    def get_status(self):
        """Get server status"""
        status = {
            'running': self.running,
            'model_loaded': self.model is not None,
            'predictions_made': self.prediction_count,
            'eeg_connected': self.eeg_acq.running if self.eeg_acq else False
        }
        
        if self.inference_times:
            status['avg_inference_time_ms'] = np.mean(self.inference_times[-100:])
            status['max_inference_time_ms'] = np.max(self.inference_times[-100:])
            
        if self.eeg_acq:
            status['eeg_buffer'] = self.eeg_acq.get_buffer_status()
            
        status['p300_status'] = self.p300_processor.get_status()
        
        return status

def main():
    """Run inference server"""
    logging.basicConfig(level=logging.INFO)
    
    server = NAIInferenceServer()
    
    try:
        if server.start_server():
            logger.info("Server running. Press Ctrl+C to stop...")
            
            # Monitor server status
            while True:
                time.sleep(10)
                status = server.get_status()
                logger.info(f"Status: {status['predictions_made']} predictions, "
                          f"EEG: {status['eeg_connected']}")
                          
    except KeyboardInterrupt:
        logger.info("Stopping server...")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        server.stop_server()

if __name__ == "__main__":
    main()