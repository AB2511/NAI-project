"""
Oddball Experiment Protocol (P300 + Fatigue)
Implements the exact experimental protocol for P300 amplitude, latency, and fatigue trends
"""

import numpy as np
import time
import threading
import logging
from pylsl import StreamOutlet, StreamInfo
import sounddevice as sd
from collections import deque
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class OddballExperiment:
    def __init__(self, total_trials=400, oddball_probability=0.2, 
                 stimulus_duration=0.2, isi=0.8, fs=44100):
        """
        Initialize oddball experiment
        
        Args:
            total_trials: Total number of trials (default 400)
            oddball_probability: Probability of oddball stimulus (default 0.2)
            stimulus_duration: Duration of each stimulus in seconds (default 0.2)
            isi: Inter-stimulus interval in seconds (default 0.8)
            fs: Audio sampling frequency (default 44100)
        """
        self.total_trials = total_trials
        self.oddball_probability = oddball_probability
        self.stimulus_duration = stimulus_duration
        self.isi = isi
        self.fs = fs
        
        # Stimulus parameters
        self.standard_freq = 440  # Hz (440 Hz "beep")
        self.oddball_freq = 880   # Hz (880 Hz "boop")
        
        # Experiment state
        self.running = False
        self.current_trial = 0
        self.trial_sequence = []
        self.results = []
        
        # LSL marker outlet
        self.marker_outlet = None
        
        # Audio generation
        self.standard_tone = self._generate_tone(self.standard_freq)
        self.oddball_tone = self._generate_tone(self.oddball_freq)
        
        # Timing
        self.experiment_start_time = None
        self.trial_times = []
        
    def _generate_tone(self, frequency):
        """Generate audio tone"""
        t = np.linspace(0, self.stimulus_duration, 
                       int(self.fs * self.stimulus_duration))
        
        # Generate sine wave with envelope to avoid clicks
        tone = 0.3 * np.sin(2 * np.pi * frequency * t)
        
        # Apply fade in/out envelope
        fade_samples = int(0.01 * self.fs)  # 10ms fade
        tone[:fade_samples] *= np.linspace(0, 1, fade_samples)
        tone[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return tone
        
    def setup_lsl_outlet(self):
        """Setup LSL marker outlet for experiment"""
        try:
            info = StreamInfo(
                name='Oddball_Markers',
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id='oddball_exp_001'
            )
            
            self.marker_outlet = StreamOutlet(info)
            logger.info("LSL marker outlet created for oddball experiment")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create LSL outlet: {e}")
            return False
            
    def generate_trial_sequence(self):
        """Generate randomized trial sequence"""
        # Calculate number of oddball trials
        n_oddball = int(self.total_trials * self.oddball_probability)
        n_standard = self.total_trials - n_oddball
        
        # Create sequence
        sequence = ['standard'] * n_standard + ['oddball'] * n_oddball
        
        # Randomize
        np.random.shuffle(sequence)
        
        # Ensure no more than 3 consecutive oddballs
        self._balance_sequence(sequence)
        
        self.trial_sequence = sequence
        
        logger.info(f"Generated sequence: {n_standard} standard, {n_oddball} oddball trials")
        return sequence
        
    def _balance_sequence(self, sequence):
        """Balance sequence to avoid too many consecutive oddballs"""
        max_consecutive = 3
        
        for i in range(len(sequence) - max_consecutive):
            # Check for too many consecutive oddballs
            if all(sequence[j] == 'oddball' for j in range(i, i + max_consecutive + 1)):
                # Find next standard to swap
                for j in range(i + max_consecutive + 1, len(sequence)):
                    if sequence[j] == 'standard':
                        sequence[i + max_consecutive], sequence[j] = sequence[j], sequence[i + max_consecutive]
                        break
                        
    def start_experiment(self):
        """Start the oddball experiment"""
        if not self.setup_lsl_outlet():
            logger.error("Failed to setup LSL outlet")
            return False
            
        if not self.trial_sequence:
            self.generate_trial_sequence()
            
        self.running = True
        self.current_trial = 0
        self.experiment_start_time = time.time()
        self.results = []
        self.trial_times = []
        
        # Send experiment start marker
        self.marker_outlet.push_sample(['EXP_START'])
        
        logger.info(f"Starting oddball experiment: {self.total_trials} trials")
        logger.info(f"Estimated duration: {self._estimate_duration():.1f} minutes")
        
        # Run experiment in separate thread
        self.experiment_thread = threading.Thread(target=self._run_experiment)
        self.experiment_thread.daemon = True
        self.experiment_thread.start()
        
        return True
        
    def _estimate_duration(self):
        """Estimate experiment duration in minutes"""
        trial_duration = self.stimulus_duration + self.isi
        total_seconds = self.total_trials * trial_duration
        return total_seconds / 60
        
    def _run_experiment(self):
        """Main experiment loop"""
        try:
            for trial_idx in range(self.total_trials):
                if not self.running:
                    break
                    
                self.current_trial = trial_idx + 1
                trial_type = self.trial_sequence[trial_idx]
                
                # Record trial start time
                trial_start = time.time()
                self.trial_times.append(trial_start)
                
                # Present stimulus
                self._present_stimulus(trial_type, trial_idx)
                
                # Inter-stimulus interval
                time.sleep(self.isi)
                
                # Progress logging
                if (trial_idx + 1) % 50 == 0:
                    progress = (trial_idx + 1) / self.total_trials * 100
                    elapsed = time.time() - self.experiment_start_time
                    logger.info(f"Progress: {progress:.1f}% ({trial_idx + 1}/{self.total_trials}) "
                               f"- Elapsed: {elapsed/60:.1f}min")
                               
            # Experiment completed
            self._finish_experiment()
            
        except Exception as e:
            logger.error(f"Experiment error: {e}")
            self.running = False
            
    def _present_stimulus(self, trial_type, trial_idx):
        """Present stimulus and send markers"""
        # Select tone
        tone = self.oddball_tone if trial_type == 'oddball' else self.standard_tone
        
        # Send pre-stimulus marker
        marker = 'TARGET' if trial_type == 'oddball' else 'NONTARGET'
        stimulus_time = time.time()
        
        # Send LSL marker
        self.marker_outlet.push_sample([marker])
        
        # Play audio stimulus
        try:
            sd.play(tone, self.fs)
            sd.wait()  # Wait for playback to complete
        except Exception as e:
            logger.warning(f"Audio playback error: {e}")
            
        # Record trial result
        trial_result = {
            'trial_number': trial_idx + 1,
            'trial_type': trial_type,
            'marker': marker,
            'stimulus_time': stimulus_time,
            'relative_time': stimulus_time - self.experiment_start_time
        }
        
        self.results.append(trial_result)
        
        logger.debug(f"Trial {trial_idx + 1}: {trial_type} ({marker})")
        
    def stop_experiment(self):
        """Stop the experiment"""
        if self.running:
            self.running = False
            logger.info("Stopping experiment...")
            
            # Send stop marker
            if self.marker_outlet:
                self.marker_outlet.push_sample(['EXP_STOP'])
                
    def _finish_experiment(self):
        """Finish experiment and save results"""
        self.running = False
        
        # Send completion marker
        self.marker_outlet.push_sample(['EXP_COMPLETE'])
        
        # Calculate statistics
        total_duration = time.time() - self.experiment_start_time
        oddball_count = sum(1 for r in self.results if r['trial_type'] == 'oddball')
        standard_count = len(self.results) - oddball_count
        
        logger.info("Experiment completed!")
        logger.info(f"Duration: {total_duration/60:.2f} minutes")
        logger.info(f"Trials completed: {len(self.results)}/{self.total_trials}")
        logger.info(f"Standard: {standard_count}, Oddball: {oddball_count}")
        logger.info(f"Actual oddball probability: {oddball_count/len(self.results):.3f}")
        
        # Save results
        self.save_results()
        
    def save_results(self):
        """Save experiment results to file"""
        if not self.results:
            return
            
        # Create results directory
        results_dir = "data_raw/oddball_experiments"
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"oddball_exp_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data
        experiment_data = {
            'experiment_info': {
                'total_trials': self.total_trials,
                'oddball_probability': self.oddball_probability,
                'stimulus_duration': self.stimulus_duration,
                'isi': self.isi,
                'standard_freq': self.standard_freq,
                'oddball_freq': self.oddball_freq,
                'start_time': self.experiment_start_time,
                'duration_minutes': (time.time() - self.experiment_start_time) / 60
            },
            'trial_sequence': self.trial_sequence,
            'results': self.results,
            'statistics': {
                'trials_completed': len(self.results),
                'standard_count': sum(1 for r in self.results if r['trial_type'] == 'standard'),
                'oddball_count': sum(1 for r in self.results if r['trial_type'] == 'oddball'),
                'actual_oddball_prob': sum(1 for r in self.results if r['trial_type'] == 'oddball') / len(self.results)
            }
        }
        
        # Save to JSON
        try:
            with open(filepath, 'w') as f:
                json.dump(experiment_data, f, indent=2)
            logger.info(f"Results saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            
    def get_status(self):
        """Get current experiment status"""
        if not self.running:
            return {
                'status': 'stopped',
                'current_trial': 0,
                'total_trials': self.total_trials,
                'progress': 0.0,
                'elapsed_time': 0.0,
                'estimated_remaining': 0.0
            }
            
        elapsed = time.time() - self.experiment_start_time if self.experiment_start_time else 0
        progress = self.current_trial / self.total_trials
        estimated_total = elapsed / progress if progress > 0 else 0
        estimated_remaining = max(0, estimated_total - elapsed)
        
        return {
            'status': 'running',
            'current_trial': self.current_trial,
            'total_trials': self.total_trials,
            'progress': progress,
            'elapsed_time': elapsed,
            'estimated_remaining': estimated_remaining,
            'current_trial_type': self.trial_sequence[self.current_trial - 1] if self.current_trial > 0 else None
        }

def main():
    """Test oddball experiment"""
    logging.basicConfig(level=logging.INFO)
    
    # Create experiment with shorter parameters for testing
    experiment = OddballExperiment(
        total_trials=20,  # Short test
        oddball_probability=0.3,
        stimulus_duration=0.2,
        isi=1.0
    )
    
    print("🎯 Oddball Experiment Test")
    print("=" * 40)
    print("This will play 20 audio tones (440Hz standard, 880Hz oddball)")
    print("LSL markers will be sent for each stimulus")
    print("Press Ctrl+C to stop early")
    print("=" * 40)
    
    try:
        if experiment.start_experiment():
            # Monitor progress
            while experiment.running:
                status = experiment.get_status()
                print(f"\rProgress: {status['progress']:.1%} "
                      f"({status['current_trial']}/{status['total_trials']}) "
                      f"- {status['elapsed_time']:.1f}s elapsed", end='')
                time.sleep(0.5)
                
            print("\n✅ Experiment completed!")
            
        else:
            print("❌ Failed to start experiment")
            
    except KeyboardInterrupt:
        print("\n⏹️ Experiment stopped by user")
        experiment.stop_experiment()

if __name__ == "__main__":
    main()