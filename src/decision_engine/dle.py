"""
Decision Logic Engine (DLE)
Adaptive feedback system based on cognitive state and fatigue
"""

import numpy as np
import time
import logging
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    RELAXED = "Relaxed"
    FOCUSED = "Focused" 
    DISTRACTED = "Distracted"
    OVERLOAD = "Overload"

class InterventionType(Enum):
    NONE = "none"
    VISUAL_CUE = "visual_cue"
    BREATHING_CUE = "breathing_cue"
    MANDATORY_PAUSE = "mandatory_pause"

class DecisionLogicEngine:
    def __init__(self, 
                 overload_threshold=0.7,
                 fatigue_threshold=0.6,
                 intervention_cooldown=30.0,
                 pause_duration=60.0):
        """
        Initialize Decision Logic Engine
        
        Args:
            overload_threshold: Confidence threshold for overload state
            fatigue_threshold: Fatigue index threshold for interventions
            intervention_cooldown: Minimum seconds between interventions
            pause_duration: Duration of mandatory pause in seconds
        """
        self.overload_threshold = overload_threshold
        self.fatigue_threshold = fatigue_threshold
        self.intervention_cooldown = intervention_cooldown
        self.pause_duration = pause_duration
        
        # State tracking
        self.current_state = CognitiveState.RELAXED
        self.current_confidence = 0.0
        self.fatigue_index = 0.0
        
        # Intervention tracking
        self.last_intervention_time = 0.0
        self.intervention_count = 0
        self.pause_start_time = None
        
        # History for trend analysis
        self.state_history = []
        self.confidence_history = []
        self.max_history = 100
        
    def update_state(self, state: str, confidence: float, fatigue_index: float = 0.0) -> Dict:
        """
        Update cognitive state and determine intervention
        
        Args:
            state: Current cognitive state string
            confidence: Model confidence (0-1)
            fatigue_index: P300-based fatigue index (0-1)
            
        Returns:
            Dict with intervention decision and metadata
        """
        current_time = time.time()
        
        # Update state
        try:
            self.current_state = CognitiveState(state)
        except ValueError:
            logger.warning(f"Unknown state: {state}, defaulting to Relaxed")
            self.current_state = CognitiveState.RELAXED
            
        self.current_confidence = confidence
        self.fatigue_index = fatigue_index
        
        # Update history
        self.state_history.append((current_time, self.current_state))
        self.confidence_history.append((current_time, confidence))
        
        # Trim history
        if len(self.state_history) > self.max_history:
            self.state_history.pop(0)
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
            
        # Check if in mandatory pause
        if self.pause_start_time is not None:
            if current_time - self.pause_start_time < self.pause_duration:
                return {
                    'intervention': InterventionType.MANDATORY_PAUSE,
                    'message': f"Mandatory pause: {self.pause_duration - (current_time - self.pause_start_time):.0f}s remaining",
                    'pause_remaining': self.pause_duration - (current_time - self.pause_start_time),
                    'state': state,
                    'confidence': confidence,
                    'fatigue_index': fatigue_index
                }
            else:
                # Pause completed
                self.pause_start_time = None
                logger.info("Mandatory pause completed")
        
        # Determine intervention
        intervention = self._determine_intervention(current_time)
        
        return {
            'intervention': intervention,
            'message': self._get_intervention_message(intervention),
            'state': state,
            'confidence': confidence,
            'fatigue_index': fatigue_index,
            'intervention_count': self.intervention_count,
            'time_since_last_intervention': current_time - self.last_intervention_time
        }
        
    def _determine_intervention(self, current_time: float) -> InterventionType:
        """Determine appropriate intervention based on current state"""
        
        # Check cooldown
        if current_time - self.last_intervention_time < self.intervention_cooldown:
            return InterventionType.NONE
            
        # High fatigue + overload = mandatory pause
        if (self.fatigue_index > self.fatigue_threshold and 
            self.current_state == CognitiveState.OVERLOAD and
            self.current_confidence > self.overload_threshold):
            
            self._trigger_intervention(current_time, InterventionType.MANDATORY_PAUSE)
            self.pause_start_time = current_time
            return InterventionType.MANDATORY_PAUSE
            
        # High confidence overload = breathing cue
        elif (self.current_state == CognitiveState.OVERLOAD and 
              self.current_confidence > self.overload_threshold):
            
            self._trigger_intervention(current_time, InterventionType.BREATHING_CUE)
            return InterventionType.BREATHING_CUE
            
        # Moderate fatigue or distraction = visual cue
        elif (self.fatigue_index > self.fatigue_threshold * 0.7 or
              (self.current_state == CognitiveState.DISTRACTED and 
               self.current_confidence > 0.6)):
            
            self._trigger_intervention(current_time, InterventionType.VISUAL_CUE)
            return InterventionType.VISUAL_CUE
            
        return InterventionType.NONE
        
    def _trigger_intervention(self, current_time: float, intervention: InterventionType):
        """Record intervention trigger"""
        self.last_intervention_time = current_time
        self.intervention_count += 1
        logger.info(f"Triggered intervention: {intervention.value}")
        
    def _get_intervention_message(self, intervention: InterventionType) -> str:
        """Get user-friendly intervention message"""
        messages = {
            InterventionType.NONE: "Continue working",
            InterventionType.VISUAL_CUE: "💡 Take a moment to refocus",
            InterventionType.BREATHING_CUE: "🫁 Take 3 deep breaths to reduce stress",
            InterventionType.MANDATORY_PAUSE: "⏸️ Mandatory break - step away from the task"
        }
        return messages.get(intervention, "Unknown intervention")
        
    def get_trend_analysis(self) -> Dict:
        """Analyze recent trends in cognitive state"""
        if len(self.state_history) < 5:
            return {'trend': 'insufficient_data'}
            
        recent_states = [s[1] for s in self.state_history[-10:]]
        recent_confidences = [c[1] for c in self.confidence_history[-10:]]
        
        # Count overload episodes
        overload_count = sum(1 for s in recent_states if s == CognitiveState.OVERLOAD)
        
        # Confidence trend
        if len(recent_confidences) >= 5:
            confidence_trend = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
        else:
            confidence_trend = 0
            
        return {
            'trend': 'improving' if confidence_trend > 0.01 else 'declining' if confidence_trend < -0.01 else 'stable',
            'overload_episodes': overload_count,
            'avg_confidence': np.mean(recent_confidences),
            'fatigue_level': 'high' if self.fatigue_index > 0.7 else 'moderate' if self.fatigue_index > 0.4 else 'low'
        }
        
    def reset_session(self):
        """Reset for new session"""
        self.state_history.clear()
        self.confidence_history.clear()
        self.intervention_count = 0
        self.last_intervention_time = 0.0
        self.pause_start_time = None
        logger.info("DLE session reset")
        
    def get_status(self) -> Dict:
        """Get current DLE status"""
        return {
            'current_state': self.current_state.value,
            'confidence': self.current_confidence,
            'fatigue_index': self.fatigue_index,
            'intervention_count': self.intervention_count,
            'in_pause': self.pause_start_time is not None,
            'trend_analysis': self.get_trend_analysis()
        }