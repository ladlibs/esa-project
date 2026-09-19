import numpy as np

class SharpTurnWarningSystem:
    def __init__(self, enter_threshold=300, exit_threshold=450, smoothing_window=8, debounce_frames=5):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.smoothing_window = smoothing_window
        self.debounce_frames = debounce_frames
        
        self.curverad_buffer = []
        self.is_warning_active = False
        self.consecutive_frames = 0
        self.last_state_target = False

    def update(self, curverad_or_none):
        """
        Call once per frame with the current frame's avg_curverad
        (or None if detection failed). Returns True/False: whether the
        warning should currently be displayed.
        """
        if curverad_or_none is not None:
            self.curverad_buffer.append(curverad_or_none)
            if len(self.curverad_buffer) > self.smoothing_window:
                self.curverad_buffer.pop(0)
                
        if not self.curverad_buffer:
            return self.is_warning_active
            
        # Use median for smoothing to be robust against outliers
        smoothed_curverad = np.median(self.curverad_buffer)
        
        # Determine target state based on hysteresis thresholds
        if self.is_warning_active:
            # If warning is on, only turn off if we go ABOVE exit_threshold
            target_state = smoothed_curverad <= self.exit_threshold
        else:
            # If warning is off, only turn on if we go BELOW enter_threshold
            target_state = smoothed_curverad < self.enter_threshold
            
        # Debounce logic
        if target_state == self.last_state_target:
            self.consecutive_frames += 1
        else:
            self.consecutive_frames = 1
            self.last_state_target = target_state
            
        # Apply state change if held for N frames
        if self.consecutive_frames >= self.debounce_frames:
            self.is_warning_active = target_state
            
        return self.is_warning_active
