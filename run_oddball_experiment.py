#!/usr/bin/env python3
"""
Oddball Experiment Runner
Runs the complete P300 oddball experiment with real-time monitoring
"""

import sys
import os
import time
import threading
import logging
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from experiments.oddball_experiment import OddballExperiment
from p300.p300_online import P300OnlineProcessor
from acquisition.lsl_acquire import LSLAcquisition
from pylsl import StreamOutlet, StreamInfo
import numpy as np

logger = logging.getLogger(__name__)

class OddballExperimentRunner:
    def __init__(self, config=None):
        """Initialize oddball experiment runner"""
        self.config = config or self._default_config()
        
        # Components
        self.experiment = None
        self.p300_processor = None
        self.eeg_acquisition = None
        
        # Data collection
        self.p300_outlet = None
        self.experiment_data = []
        
        # Status
        self.running = False
        
    def _default_config(self):
        """Default experiment configuration"""
        return {
            'total_trials': 400,
            'oddball_probability': 0.2,
            'stimulus_duration': 0.2,
            'isi': 0.8,
            'fs_eeg': 256,
            'fs_audio': 44100,
            'channels': ['Fp1', 'Fp2', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
        }
        
    def setup_components(self):
        """Setup all experiment components"""
        logger.info("Setting up oddball experiment components...")
        
        # 1. Setup oddball experiment
        self.experiment = OddballExperiment(
            total_trials=self.config['total_trials'],
            oddball_probability=self.config['oddball_probability'],
            stimulus_duration=self.config['stimulus_duration'],
            isi=self.config['isi'],
            fs=self.config['fs_audio']
        )
        
        # 2. Setup P300 processor
        self.p300_processor = P300OnlineProcessor(
            fs=self.config['fs_eeg'],
            epoch_window=(-0.2, 0.8),
            baseline_window=(-0.2, 0.0)
        )
        
        # Connect to marker stream
        if not self.p300_processor.connect_marker_stream(timeout=10):
            logger.warning("No marker stream found - P300 processing may not work")
            
        # 3. Setup EEG acquisition (optional - for real EEG)
        try:
            self.eeg_acquisition = LSLAcquisition()
            if self.eeg_acquisition.connect():
                logger.info("EEG acquisition connected")
                # Set channel info for P300 processor
                channel_names = self.eeg_acquisition.get_channel_names()
                self.p300_processor.set_channel_info(channel_names)
            else:
                logger.warning("No EEG stream found - using synthetic data")
                self.eeg_acquisition = None
        except Exception as e:
            logger.warning(f"EEG acquisition setup failed: {e}")
            self.eeg_acquisition = None
            
        # 4. Setup P300 data outlet for dashboard
        self._setup_p300_outlet()
        
        return True
        
    def _setup_p300_outlet(self):
        """Setup LSL outlet for P300 data"""
        try:
            info = StreamInfo(
                name='NAI_P300',
                type='P300',
                channel_count=4,  # amplitude, latency, running_amp, fatigue
                nominal_srate=0,
                channel_format='float32',
                source_id='oddball_p300_001'
            )
            
            # Add channel labels
            channels = info.desc().append_child("channels")
            for label in ['amplitude', 'latency', 'running_amplitude', 'fatigue_index']:
                channels.append_child("channel").append_child_value("label", label)
                
            self.p300_outlet = StreamOutlet(info)
            logger.info("P300 data outlet created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create P300 outlet: {e}")
            return False
            
    def run_experiment(self):
        """Run the complete oddball experiment"""
        logger.info("🎯 Starting Oddball Experiment")
        logger.info("=" * 50)
        
        if not self.setup_components():
            logger.error("Failed to setup components")
            return False
            
        # Start EEG acquisition if available
        if self.eeg_acquisition:
            self.eeg_acquisition.start_acquisition()
            
        # Start P300 processing thread
        self.running = True
        p300_thread = threading.Thread(target=self._p300_processing_loop)
        p300_thread.daemon = True
        p300_thread.start()
        
        # Start experiment
        if not self.experiment.start_experiment():
            logger.error("Failed to start experiment")
            return False
            
        logger.info("Experiment started! Monitor progress...")
        logger.info("Dashboard: http://localhost:8501")
        logger.info("Press Ctrl+C to stop early")
        
        try:
            # Monitor experiment progress
            while self.experiment.running:
                status = self.experiment.get_status()
                
                # Print progress every 10 seconds
                if int(status['elapsed_time']) % 10 == 0:
                    logger.info(f"Progress: {status['progress']:.1%} "
                               f"({status['current_trial']}/{status['total_trials']}) "
                               f"- {status['elapsed_time']/60:.1f}min elapsed, "
                               f"{status['estimated_remaining']/60:.1f}min remaining")
                    
                time.sleep(1)
                
            # Experiment completed
            logger.info("✅ Experiment completed successfully!")
            self._print_results()
            
        except KeyboardInterrupt:
            logger.info("⏹️ Experiment stopped by user")
            self.experiment.stop_experiment()
            
        finally:
            self.running = False
            if self.eeg_acquisition:
                self.eeg_acquisition.stop_acquisition()
                
        return True
        
    def _p300_processing_loop(self):
        """P300 processing loop"""
        logger.info("P300 processing started")
        
        while self.running:
            try:
                # Update EEG buffer if acquisition is running
                if self.eeg_acquisition and self.eeg_acquisition.is_running():
                    eeg_data, timestamps = self.eeg_acquisition.get_recent_data(duration=1.0)
                    if eeg_data is not None:
                        self.p300_processor.update_eeg_buffer(eeg_data, timestamps)
                        
                # Process markers and extract P300
                p300_result = self.p300_processor.process_markers()
                
                if p300_result:
                    # Store result
                    self.experiment_data.append(p300_result)
                    
                    # Send to dashboard via LSL
                    if self.p300_outlet:
                        sample = [
                            p300_result['amplitude'],
                            p300_result['latency'],
                            p300_result['running_amplitude'],
                            p300_result['fatigue_index']
                        ]
                        self.p300_outlet.push_sample(sample)
                        
                    # Log significant events
                    if p300_result['fatigue_index'] > 0.5:
                        logger.warning(f"High fatigue detected: {p300_result['fatigue_index']:.3f}")
                        
                time.sleep(0.01)  # 100 Hz processing rate
                
            except Exception as e:
                logger.error(f"P300 processing error: {e}")
                time.sleep(0.1)
                
        logger.info("P300 processing stopped")
        
    def _print_results(self):
        """Print experiment results summary"""
        if not self.experiment_data:
            logger.warning("No P300 data collected")
            return
            
        # Calculate statistics
        amplitudes = [d['amplitude'] for d in self.experiment_data]
        latencies = [d['latency'] for d in self.experiment_data]
        fatigue_indices = [d['fatigue_index'] for d in self.experiment_data]
        
        logger.info("\n📊 Experiment Results Summary")
        logger.info("-" * 40)
        logger.info(f"P300 Events Detected: {len(self.experiment_data)}")
        logger.info(f"Average Amplitude: {np.mean(amplitudes):.2f} ± {np.std(amplitudes):.2f} µV")
        logger.info(f"Average Latency: {np.mean(latencies):.1f} ± {np.std(latencies):.1f} ms")
        logger.info(f"Final Fatigue Index: {fatigue_indices[-1]:.3f}")
        logger.info(f"Peak Fatigue Index: {max(fatigue_indices):.3f}")
        
        # Fatigue trend
        if len(fatigue_indices) > 10:
            early_fatigue = np.mean(fatigue_indices[:10])
            late_fatigue = np.mean(fatigue_indices[-10:])
            fatigue_change = late_fatigue - early_fatigue
            logger.info(f"Fatigue Change: {fatigue_change:+.3f} (early: {early_fatigue:.3f}, late: {late_fatigue:.3f})")
            
        # Amplitude decline
        if len(amplitudes) > 20:
            early_amp = np.mean(amplitudes[:20])
            late_amp = np.mean(amplitudes[-20:])
            amp_decline = (early_amp - late_amp) / early_amp * 100
            logger.info(f"Amplitude Decline: {amp_decline:.1f}% (early: {early_amp:.2f}, late: {late_amp:.2f})")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run P300 Oddball Experiment')
    parser.add_argument('--trials', type=int, default=400, help='Total number of trials')
    parser.add_argument('--oddball-prob', type=float, default=0.2, help='Oddball probability')
    parser.add_argument('--isi', type=float, default=0.8, help='Inter-stimulus interval (s)')
    parser.add_argument('--duration', type=float, default=0.2, help='Stimulus duration (s)')
    parser.add_argument('--demo', action='store_true', help='Run short demo (20 trials)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'total_trials': 20 if args.demo else args.trials,
        'oddball_probability': args.oddball_prob,
        'stimulus_duration': args.duration,
        'isi': args.isi,
        'fs_eeg': 256,
        'fs_audio': 44100,
        'channels': ['Fp1', 'Fp2', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
    }
    
    # Create and run experiment
    runner = OddballExperimentRunner(config)
    
    if args.demo:
        logger.info("🎯 Running DEMO mode (20 trials, ~30 seconds)")
    else:
        duration_min = (config['total_trials'] * (config['stimulus_duration'] + config['isi'])) / 60
        logger.info(f"🎯 Running full experiment ({config['total_trials']} trials, ~{duration_min:.1f} minutes)")
        
    logger.info("Make sure to:")
    logger.info("1. Start the dashboard: streamlit run src/dashboard/app.py")
    logger.info("2. Connect EEG system (optional)")
    logger.info("3. Ensure audio is working")
    logger.info("")
    
    success = runner.run_experiment()
    
    if success:
        logger.info("🎉 Experiment completed successfully!")
        logger.info("Data saved to: data_raw/oddball_experiments/")
        logger.info("Check the dashboard for real-time results")
    else:
        logger.error("❌ Experiment failed")
        sys.exit(1)

if __name__ == "__main__":
    main()