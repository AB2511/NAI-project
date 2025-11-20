/*
 * Arduino Timing Module (ATM) for NeuroAdaptive Interface
 * Provides millisecond-precise event markers via LSL
 * 
 * Hardware: Arduino Uno/Nano with USB connection
 * Purpose: Send precise timing markers for P300 experiments
 * 
 * Commands:
 * - 'S': Start oddball sequence
 * - 'T': Send target stimulus marker
 * - 'N': Send non-target stimulus marker  
 * - 'R': Reset/stop sequence
 * - 'C': Calibration mode
 */

// Pin definitions
const int LED_PIN = 13;        // Built-in LED for visual feedback
const int BUTTON_PIN = 2;      // Button for manual triggers
const int TRIGGER_OUT_PIN = 3; // Digital output for external triggers

// Timing parameters
const unsigned long STIMULUS_INTERVAL = 1000; // 1 second between stimuli
const unsigned long STIMULUS_DURATION = 100;  // 100ms stimulus duration
const int TARGET_PROBABILITY = 20;             // 20% target probability

// State variables
bool sequenceActive = false;
bool stimulusActive = false;
unsigned long lastStimulusTime = 0;
unsigned long stimulusStartTime = 0;
int stimulusCount = 0;
int targetCount = 0;

// Marker codes
const char MARKER_TARGET = 'T';
const char MARKER_NONTARGET = 'N';
const char MARKER_START = 'S';
const char MARKER_STOP = 'E';

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(TRIGGER_OUT_PIN, OUTPUT);
  
  // Initial state
  digitalWrite(LED_PIN, LOW);
  digitalWrite(TRIGGER_OUT_PIN, LOW);
  
  // Seed random number generator
  randomSeed(analogRead(0));
  
  // Startup message
  Serial.println("NAI-ATM-READY");
  Serial.println("Commands: S=Start, R=Reset, C=Calibration, T=Target, N=NonTarget");
  
  // Flash LED to indicate ready
  for(int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();
    handleCommand(command);
  }
  
  // Check button press
  if (digitalRead(BUTTON_PIN) == LOW) {
    delay(50); // Debounce
    if (digitalRead(BUTTON_PIN) == LOW) {
      sendManualTrigger();
      while(digitalRead(BUTTON_PIN) == LOW); // Wait for release
    }
  }
  
  // Handle automatic sequence
  if (sequenceActive) {
    handleSequence();
  }
  
  // Handle stimulus timing
  if (stimulusActive) {
    handleStimulus();
  }
}

void handleCommand(char command) {
  switch(command) {
    case 'S':
    case 's':
      startSequence();
      break;
      
    case 'R':
    case 'r':
      resetSequence();
      break;
      
    case 'C':
    case 'c':
      calibrationMode();
      break;
      
    case 'T':
    case 't':
      sendTargetMarker();
      break;
      
    case 'N':
    case 'n':
      sendNonTargetMarker();
      break;
      
    default:
      Serial.println("Unknown command");
      break;
  }
}

void startSequence() {
  sequenceActive = true;
  stimulusCount = 0;
  targetCount = 0;
  lastStimulusTime = millis();
  
  Serial.println("SEQUENCE_START");
  sendMarker(MARKER_START);
  
  digitalWrite(LED_PIN, HIGH);
  delay(100);
  digitalWrite(LED_PIN, LOW);
}

void resetSequence() {
  sequenceActive = false;
  stimulusActive = false;
  
  Serial.println("SEQUENCE_STOP");
  sendMarker(MARKER_STOP);
  
  digitalWrite(LED_PIN, LOW);
  digitalWrite(TRIGGER_OUT_PIN, LOW);
  
  // Print statistics
  Serial.print("Total stimuli: ");
  Serial.println(stimulusCount);
  Serial.print("Target stimuli: ");
  Serial.println(targetCount);
  Serial.print("Target percentage: ");
  if (stimulusCount > 0) {
    Serial.println((targetCount * 100) / stimulusCount);
  } else {
    Serial.println(0);
  }
}

void handleSequence() {
  unsigned long currentTime = millis();
  
  // Check if it's time for next stimulus
  if (currentTime - lastStimulusTime >= STIMULUS_INTERVAL) {
    // Determine stimulus type (target or non-target)
    bool isTarget = (random(100) < TARGET_PROBABILITY);
    
    if (isTarget) {
      sendTargetMarker();
      targetCount++;
    } else {
      sendNonTargetMarker();
    }
    
    stimulusCount++;
    lastStimulusTime = currentTime;
    
    // Start stimulus presentation
    stimulusActive = true;
    stimulusStartTime = currentTime;
    digitalWrite(TRIGGER_OUT_PIN, HIGH);
    digitalWrite(LED_PIN, HIGH);
  }
}

void handleStimulus() {
  unsigned long currentTime = millis();
  
  // Check if stimulus duration is over
  if (currentTime - stimulusStartTime >= STIMULUS_DURATION) {
    stimulusActive = false;
    digitalWrite(TRIGGER_OUT_PIN, LOW);
    digitalWrite(LED_PIN, LOW);
  }
}

void sendTargetMarker() {
  sendMarker(MARKER_TARGET);
  Serial.println("TARGET");
}

void sendNonTargetMarker() {
  sendMarker(MARKER_NONTARGET);
  Serial.println("NONTARGET");
}

void sendManualTrigger() {
  sendMarker('M');
  Serial.println("MANUAL_TRIGGER");
  
  // Brief LED flash
  digitalWrite(LED_PIN, HIGH);
  delay(50);
  digitalWrite(LED_PIN, LOW);
}

void sendMarker(char marker) {
  // Send marker with precise timestamp
  unsigned long timestamp = micros();
  
  Serial.print("MARKER:");
  Serial.print(marker);
  Serial.print(":");
  Serial.println(timestamp);
  
  // Brief trigger pulse
  digitalWrite(TRIGGER_OUT_PIN, HIGH);
  delayMicroseconds(100);
  digitalWrite(TRIGGER_OUT_PIN, LOW);
}

void calibrationMode() {
  Serial.println("CALIBRATION_MODE");
  
  // Send 10 test markers with precise timing
  for(int i = 0; i < 10; i++) {
    Serial.print("CAL_MARKER_");
    Serial.println(i);
    sendMarker('C');
    
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(900);
  }
  
  Serial.println("CALIBRATION_COMPLETE");
}

/*
 * LSL Integration Notes:
 * 
 * To use with LSL, create a Python script that:
 * 1. Connects to Arduino serial port
 * 2. Parses marker messages
 * 3. Sends markers to LSL stream
 * 
 * Example Python integration:
 * 
 * import serial
 * from pylsl import StreamOutlet, StreamInfo
 * 
 * # Setup LSL outlet
 * info = StreamInfo('Arduino_Markers', 'Markers', 1, 0, 'string', 'arduino001')
 * outlet = StreamOutlet(info)
 * 
 * # Connect to Arduino
 * ser = serial.Serial('COM3', 115200)  # Adjust port
 * 
 * while True:
 *     line = ser.readline().decode().strip()
 *     if line.startswith('MARKER:'):
 *         parts = line.split(':')
 *         marker = parts[1]
 *         timestamp = int(parts[2])
 *         outlet.push_sample([marker], timestamp/1000000.0)
 */