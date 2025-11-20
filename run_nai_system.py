#!/usr/bin/env python3
"""
NeuroAdaptive Interface (NAI) System Launcher
Complete system orchestration and management
"""

import subprocess
import time
import sys
import os
import signal
import logging
from pathlib import Path
import argparse
from multiprocessing import Process
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NAI-System')

class NAISystemManager:
    def __init__(self, demo_mode=False, arduino_port=None):
        self.demo_mode = demo_mode
        self.arduino_port = arduino_port
        self.processes = {}
        self.running = False
        
        # Project paths
        self.project_root = Path(__file__).parent
        self.src_path = self.project_root / 'src'
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'numpy', 'pandas', 'mne', 'scipy', 'scikit-learn',
            'pylsl', 'streamlit', 'matplotlib', 'seaborn', 'joblib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            logger.error(f"❌ Missing packages: {missing_packages}")
            logger.info("Install with: pip install -r requirements.txt")
            return False
            
        logger.info("✅ All dependencies satisfied")
        return True
        
    def check_model(self):
        """Check if trained model exists"""
        model_path = self.project_root / 'models' / 'nai_voting_model.pkl'
        
        if not model_path.exists():
            logger.warning("⚠️ No trained model found")
            logger.info("Run calibration notebook to train a model")
            
            # Create a dummy model for demo
            if self.demo_mode:
                logger.info("🎭 Creating demo model...")
                self._create_demo_model()
                return True
            return False
            
        logger.info("✅ Trained model found")
        return True
        
    def _create_demo_model(self):
        """Create a simple demo model for testing"""
        try:
            import numpy as np
            from sklearn.ensemble import VotingClassifier, RandomForestClassifier
            from sklearn.svm import LinearSVC
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            # Create dummy training data
            np.random.seed(42)
            X = np.random.randn(400, 25)  # 25 features
            y = np.random.choice(['Relaxed', 'Focused', 'Distracted', 'Overload'], 400)
            
            # Create and train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            svm = LinearSVC(max_iter=1000, random_state=42)
            model = VotingClassifier([('rf', rf), ('svm', svm)], voting='soft')
            model.fit(X_scaled, y)
            
            # Save model
            model_dir = self.project_root / 'models'
            model_dir.mkdir(exist_ok=True)
            
            model_data = {
                'model': model,
                'scaler': scaler,
                'demo_mode': True
            }
            
            joblib.dump(model_data, model_dir / 'nai_voting_model.pkl')
            logger.info("✅ Demo model created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create demo model: {e}")
            
    def start_arduino_bridge(self):
        """Start Arduino-LSL bridge if Arduino port specified"""
        if not self.arduino_port:
            return True
            
        logger.info(f"🔌 Starting Arduino bridge on {self.arduino_port}")
        
        try:
            cmd = [
                sys.executable,
                str(self.src_path / 'atm' / 'arduino_lsl_bridge.py'),
                '--port', self.arduino_port
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            
            self.processes['arduino'] = process
            time.sleep(2)  # Give it time to start
            
            if process.poll() is None:
                logger.info("✅ Arduino bridge started")
                return True
            else:
                logger.error("❌ Arduino bridge failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start Arduino bridge: {e}")
            return False
            
    def start_inference_server(self):
        """Start ML inference server"""
        logger.info("🤖 Starting inference server...")
        
        try:
            cmd = [
                sys.executable,
                str(self.src_path / 'inference' / 'infer_server.py')
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            
            self.processes['inference'] = process
            time.sleep(3)  # Give it time to start
            
            if process.poll() is None:
                logger.info("✅ Inference server started")
                return True
            else:
                logger.error("❌ Inference server failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start inference server: {e}")
            return False
            
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        logger.info("📊 Starting dashboard...")
        
        try:
            cmd = [
                'streamlit', 'run',
                str(self.src_path / 'dashboard' / 'app.py'),
                '--server.headless', 'true',
                '--server.port', '8501'
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root
            )
            
            self.processes['dashboard'] = process
            time.sleep(5)  # Give Streamlit time to start
            
            if process.poll() is None:
                logger.info("✅ Dashboard started at http://localhost:8501")
                return True
            else:
                logger.error("❌ Dashboard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard: {e}")
            return False
            
    def start_system(self):
        """Start the complete NAI system"""
        logger.info("🚀 Starting NeuroAdaptive Interface System")
        
        # Check prerequisites
        if not self.check_dependencies():
            return False
            
        if not self.check_model():
            return False
            
        # Start components in order
        success = True
        
        # 1. Arduino bridge (optional)
        if self.arduino_port:
            success &= self.start_arduino_bridge()
            
        # 2. Inference server
        if success:
            success &= self.start_inference_server()
            
        # 3. Dashboard
        if success:
            success &= self.start_dashboard()
            
        if success:
            self.running = True
            logger.info("🎉 NAI System started successfully!")
            logger.info("📊 Dashboard: http://localhost:8501")
            
            if self.demo_mode:
                logger.info("🎭 Running in DEMO mode with synthetic data")
            else:
                logger.info("🧠 Connect your EEG device and start streaming via LSL")
                
            return True
        else:
            logger.error("❌ System startup failed")
            self.stop_system()
            return False
            
    def stop_system(self):
        """Stop all system components"""
        logger.info("⏹️ Stopping NAI System...")
        
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name}...")
                
                # Try graceful shutdown first
                process.terminate()
                
                # Wait a bit for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    process.kill()
                    process.wait()
                    
                logger.info(f"✅ {name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
                
        self.processes.clear()
        self.running = False
        logger.info("✅ NAI System stopped")
        
    def get_system_status(self):
        """Get status of all system components"""
        status = {}
        
        for name, process in self.processes.items():
            if process.poll() is None:
                status[name] = "running"
            else:
                status[name] = "stopped"
                
        return status
        
    def monitor_system(self):
        """Monitor system health"""
        logger.info("👁️ Monitoring system (Ctrl+C to stop)...")
        
        try:
            while self.running:
                status = self.get_system_status()
                
                # Check if any process died
                for name, state in status.items():
                    if state == "stopped":
                        logger.warning(f"⚠️ {name} process stopped unexpectedly")
                        
                # Log status every 30 seconds
                time.sleep(30)
                logger.info(f"System status: {status}")
                
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        finally:
            self.stop_system()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='NeuroAdaptive Interface System')
    parser.add_argument('--demo', action='store_true', 
                       help='Run in demo mode with synthetic data')
    parser.add_argument('--arduino-port', type=str,
                       help='Arduino serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Start system without monitoring')
    
    args = parser.parse_args()
    
    # Create system manager
    manager = NAISystemManager(
        demo_mode=args.demo,
        arduino_port=args.arduino_port
    )
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("🛑 Received interrupt signal")
        manager.stop_system()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start system
        if manager.start_system():
            if not args.no_monitor:
                manager.monitor_system()
            else:
                logger.info("System started. Use Ctrl+C to stop.")
                while manager.running:
                    time.sleep(1)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"System error: {e}")
        manager.stop_system()
        sys.exit(1)

if __name__ == "__main__":
    main()