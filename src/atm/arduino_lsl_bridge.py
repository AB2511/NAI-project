"""
Arduino-LSL Bridge
Connects Arduino Timing Module to LSL marker stream
"""

import serial
import time
import logging
from pylsl import StreamOutlet, StreamInfo
import threading
from queue import Queue
import argparse

logger = logging.getLogger(__name__)

class ArduinoLSLBridge:
    def __init__(self, port='COM3', baudrate=115200, lsl_name='Arduino_Markers'):
        self.port = port
        self.baudrate = baudrate
        self.lsl_name = lsl_name
        
        # Serial connection
        self.serial_conn = None
        self.connected = False
        
        # LSL outlet
        self.lsl_outlet = None
        
        # Threading
        self.running = False
        self.read_thread = None
        self.marker_queue = Queue()
        
        # Statistics
        self.markers_sent = 0
        self.start_time = None
        
    def connect_arduino(self):
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            
            # Check if Arduino is ready
            self.serial_conn.write(b'R\n')  # Reset command
            time.sleep(0.5)
            
            # Read any initial messages
            while self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode().strip()
                logger.info(f"Arduino: {line}")
                
            self.connected = True
            logger.info(f"Connected to Arduino on {self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
            
    def setup_lsl_outlet(self):
        """Setup LSL marker outlet"""
        try:
            info = StreamInfo(
                name=self.lsl_name,
                type='Markers',
                channel_count=1,
                nominal_srate=0,
                channel_format='string',
                source_id='arduino_atm_001'
            )
            
            self.lsl_outlet = StreamOutlet(info)
            logger.info(f"LSL outlet created: {self.lsl_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create LSL outlet: {e}")
            return False
            
    def start_bridge(self):
        """Start the Arduino-LSL bridge"""
        if not self.connect_arduino():
            return False
            
        if not self.setup_lsl_outlet():
            return False
            
        self.running = True
        self.start_time = time.time()
        
        # Start reading thread
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
        
        logger.info("Arduino-LSL bridge started")
        return True
        
    def stop_bridge(self):
        """Stop the bridge"""
        self.running = False
        
        if self.read_thread:
            self.read_thread.join()
            
        if self.serial_conn:
            self.serial_conn.close()
            
        logger.info("Arduino-LSL bridge stopped")
        
        # Print statistics
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Session stats: {self.markers_sent} markers in {duration:.1f}s "
                       f"({self.markers_sent/duration:.1f} markers/s)")
                       
    def _read_loop(self):
        """Main reading loop"""
        while self.running and self.connected:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode().strip()
                    self._process_line(line)
                else:
                    time.sleep(0.001)  # Small sleep to prevent CPU overload
                    
            except Exception as e:
                logger.error(f"Read loop error: {e}")
                time.sleep(0.1)
                
    def _process_line(self, line):
        """Process line from Arduino"""
        if not line:
            return
            
        logger.debug(f"Arduino: {line}")
        
        # Parse marker messages
        if line.startswith('MARKER:'):
            self._handle_marker(line)
        elif line in ['SEQUENCE_START', 'SEQUENCE_STOP']:
            self._handle_sequence_event(line)
        elif line in ['TARGET', 'NONTARGET']:
            self._handle_stimulus_event(line)
        else:
            # Log other messages
            logger.info(f"Arduino: {line}")
            
    def _handle_marker(self, line):
        """Handle marker message"""
        try:
            # Parse: MARKER:T:123456789
            parts = line.split(':')
            if len(parts) >= 3:
                marker_type = parts[1]
                timestamp_us = int(parts[2])
                
                # Convert to LSL timestamp (seconds)
                lsl_timestamp = timestamp_us / 1000000.0
                
                # Send to LSL
                self.lsl_outlet.push_sample([marker_type], lsl_timestamp)
                self.markers_sent += 1
                
                logger.debug(f"Sent marker: {marker_type} at {lsl_timestamp}")
                
        except Exception as e:
            logger.error(f"Failed to process marker: {line}, error: {e}")
            
    def _handle_sequence_event(self, event):
        """Handle sequence start/stop events"""
        marker = 'START' if event == 'SEQUENCE_START' else 'STOP'
        self.lsl_outlet.push_sample([marker])
        logger.info(f"Sequence event: {event}")
        
    def _handle_stimulus_event(self, event):
        """Handle stimulus events"""
        # These are already sent as markers, just log
        logger.debug(f"Stimulus: {event}")
        
    def send_command(self, command):
        """Send command to Arduino"""
        if self.connected and self.serial_conn:
            try:
                self.serial_conn.write(f"{command}\n".encode())
                logger.info(f"Sent command: {command}")
                return True
            except Exception as e:
                logger.error(f"Failed to send command: {e}")
                return False
        return False
        
    def start_sequence(self):
        """Start oddball sequence"""
        return self.send_command('S')
        
    def stop_sequence(self):
        """Stop sequence"""
        return self.send_command('R')
        
    def calibration_mode(self):
        """Run calibration"""
        return self.send_command('C')
        
    def get_status(self):
        """Get bridge status"""
        return {
            'connected': self.connected,
            'running': self.running,
            'markers_sent': self.markers_sent,
            'port': self.port,
            'lsl_name': self.lsl_name
        }

def main():
    """Main function for standalone operation"""
    parser = argparse.ArgumentParser(description='Arduino-LSL Bridge')
    parser.add_argument('--port', default='COM3', help='Arduino serial port')
    parser.add_argument('--baudrate', type=int, default=115200, help='Serial baudrate')
    parser.add_argument('--lsl-name', default='Arduino_Markers', help='LSL stream name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create bridge
    bridge = ArduinoLSLBridge(
        port=args.port,
        baudrate=args.baudrate,
        lsl_name=args.lsl_name
    )
    
    try:
        if bridge.start_bridge():
            logger.info("Bridge running. Commands:")
            logger.info("  's' - Start sequence")
            logger.info("  'r' - Reset/stop sequence") 
            logger.info("  'c' - Calibration mode")
            logger.info("  'q' - Quit")
            
            # Interactive mode
            while True:
                try:
                    cmd = input().strip().lower()
                    
                    if cmd == 'q':
                        break
                    elif cmd == 's':
                        bridge.start_sequence()
                    elif cmd == 'r':
                        bridge.stop_sequence()
                    elif cmd == 'c':
                        bridge.calibration_mode()
                    elif cmd == 'status':
                        status = bridge.get_status()
                        logger.info(f"Status: {status}")
                    else:
                        logger.info("Unknown command")
                        
                except EOFError:
                    break
                    
        else:
            logger.error("Failed to start bridge")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        bridge.stop_bridge()

if __name__ == "__main__":
    main()