"""
Signal processing tools for extracting event-triggered features.
Supports representation learning for decision signals.
"""

import numpy as np
from scipy import signal
from scipy.stats import zscore
from typing import List, Tuple, Dict, Optional
from collections import defaultdict


class EventTriggeredFeatureExtractor:
    """
    Extract event-triggered features from neural/behavioral signals.
    Aligns signals to behavioral events for analysis.
    """
    
    def __init__(self, 
                 window_pre: float = 0.5,
                 window_post: float = 1.0,
                 sampling_rate: float = 1000.0):
        """
        Initialize feature extractor.
        
        Args:
            window_pre: Time window before event (seconds)
            window_post: Time window after event (seconds)
            sampling_rate: Sampling rate in Hz
        """
        self.window_pre = window_pre
        self.window_post = window_post
        self.sampling_rate = sampling_rate
        
        self.n_samples_pre = int(window_pre * sampling_rate)
        self.n_samples_post = int(window_post * sampling_rate)
        self.n_samples_total = self.n_samples_pre + self.n_samples_post + 1
    
    def extract_event_triggered_average(self,
                                       signal_data: np.ndarray,
                                       event_times: np.ndarray,
                                       time_axis: Optional[np.ndarray] = None) -> Dict:
        """
        Extract event-triggered average (ETA).
        
        Args:
            signal_data: 1D or 2D signal array (time x channels)
            event_times: Array of event times in samples or seconds
            time_axis: Optional time axis for event_times (if in seconds)
        
        Returns:
            Dictionary with ETA, time axis, and statistics
        """
        if time_axis is not None:
            # Convert event times from seconds to samples
            event_indices = np.searchsorted(time_axis, event_times)
        else:
            event_indices = event_times.astype(int)
        
        # Filter valid events
        valid_events = []
        for idx in event_indices:
            if idx >= self.n_samples_pre and idx < len(signal_data) - self.n_samples_post:
                valid_events.append(idx)
        
        if len(valid_events) == 0:
            return {
                'eta': None,
                'time_axis': None,
                'n_events': 0,
                'std': None
            }
        
        # Extract windows
        if signal_data.ndim == 1:
            windows = np.zeros((len(valid_events), self.n_samples_total))
            for i, event_idx in enumerate(valid_events):
                start = event_idx - self.n_samples_pre
                end = event_idx + self.n_samples_post + 1
                windows[i] = signal_data[start:end]
        else:
            windows = np.zeros((len(valid_events), self.n_samples_total, signal_data.shape[1]))
            for i, event_idx in enumerate(valid_events):
                start = event_idx - self.n_samples_pre
                end = event_idx + self.n_samples_post + 1
                windows[i] = signal_data[start:end, :]
        
        # Compute average
        eta = np.mean(windows, axis=0)
        std = np.std(windows, axis=0)
        
        # Time axis
        time_axis_eta = np.arange(-self.n_samples_pre, self.n_samples_post + 1) / self.sampling_rate
        
        return {
            'eta': eta,
            'time_axis': time_axis_eta,
            'n_events': len(valid_events),
            'std': std,
            'individual_windows': windows
        }
    
    def extract_peri_event_features(self,
                                    signal_data: np.ndarray,
                                    event_times: np.ndarray,
                                    feature_funcs: List[callable],
                                    time_axis: Optional[np.ndarray] = None) -> Dict:
        """
        Extract multiple features around events.
        
        Args:
            signal_data: Signal array
            event_times: Event times
            feature_funcs: List of functions to compute features
            time_axis: Optional time axis
        
        Returns:
            Dictionary with features for each event
        """
        if time_axis is not None:
            event_indices = np.searchsorted(time_axis, event_times)
        else:
            event_indices = event_times.astype(int)
        
        valid_events = []
        for idx in event_indices:
            if idx >= self.n_samples_pre and idx < len(signal_data) - self.n_samples_post:
                valid_events.append(idx)
        
        features = defaultdict(list)
        
        for event_idx in valid_events:
            start = event_idx - self.n_samples_pre
            end = event_idx + self.n_samples_post + 1
            window = signal_data[start:end]
            
            for i, func in enumerate(feature_funcs):
                feature_name = func.__name__ if hasattr(func, '__name__') else f'feature_{i}'
                features[feature_name].append(func(window))
        
        return dict(features)
    
    def align_signals_to_actions(self,
                                 neural_signal: np.ndarray,
                                 action_times: np.ndarray,
                                 action_values: np.ndarray,
                                 time_axis: Optional[np.ndarray] = None) -> Dict:
        """
        Align neural signals to action events.
        
        Args:
            neural_signal: Neural signal data
            action_times: Times of actions
            action_values: Action values
            time_axis: Optional time axis
        
        Returns:
            Dictionary with aligned signals grouped by action
        """
        if time_axis is not None:
            action_indices = np.searchsorted(time_axis, action_times)
        else:
            action_indices = action_times.astype(int)
        
        aligned_by_action = defaultdict(list)
        
        for idx, action_val in zip(action_indices, action_values):
            if idx >= self.n_samples_pre and idx < len(neural_signal) - self.n_samples_post:
                start = idx - self.n_samples_pre
                end = idx + self.n_samples_post + 1
                window = neural_signal[start:end]
                aligned_by_action[action_val].append(window)
        
        # Convert to arrays
        result = {}
        for action, windows in aligned_by_action.items():
            result[action] = np.array(windows)
        
        return result


class SignalProcessor:
    """General signal processing utilities."""
    
    @staticmethod
    def bandpass_filter(data: np.ndarray,
                       lowcut: float,
                       highcut: float,
                       sampling_rate: float,
                       order: int = 4) -> np.ndarray:
        """Apply bandpass filter."""
        nyquist = sampling_rate / 2.0
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data, axis=0)
    
    @staticmethod
    def compute_power_spectrum(data: np.ndarray,
                               sampling_rate: float,
                               nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density."""
        if data.ndim == 1:
            freqs, psd = signal.welch(data, sampling_rate, nperseg=nperseg)
        else:
            freqs, psd = signal.welch(data, sampling_rate, nperseg=nperseg, axis=0)
        return freqs, psd
    
    @staticmethod
    def compute_phase_locking_value(signals: np.ndarray,
                                   freqs: np.ndarray,
                                   sampling_rate: float) -> np.ndarray:
        """Compute phase locking value across signals."""
        from scipy.signal import hilbert
        
        n_signals = signals.shape[1] if signals.ndim > 1 else 1
        
        if signals.ndim == 1:
            signals = signals[:, np.newaxis]
        
        plv_matrix = np.zeros((len(freqs), n_signals, n_signals))
        
        for i, freq in enumerate(freqs):
            # Bandpass filter around frequency
            filtered = SignalProcessor.bandpass_filter(
                signals, freq - 2, freq + 2, sampling_rate
            )
            
            # Get phase using Hilbert transform
            analytic = hilbert(filtered, axis=0)
            phases = np.angle(analytic)
            
            # Compute PLV
            for j in range(n_signals):
                for k in range(n_signals):
                    if j != k:
                        phase_diff = phases[:, j] - phases[:, k]
                        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                        plv_matrix[i, j, k] = plv
        
        return plv_matrix
    
    @staticmethod
    def zscore_normalize(data: np.ndarray, axis: int = 0) -> np.ndarray:
        """Z-score normalization."""
        return zscore(data, axis=axis)
    
    @staticmethod
    def extract_time_frequency_features(data: np.ndarray,
                                        sampling_rate: float,
                                        freqs: np.ndarray) -> np.ndarray:
        """Extract time-frequency features using wavelet transform."""
        from scipy.signal import cwt, ricker
        
        widths = sampling_rate / (2 * freqs)
        cwt_matrix = cwt(data, ricker, widths)
        return np.abs(cwt_matrix)


class DecisionSignalExtractor:
    """Extract decision-related signals from behavioral data."""
    
    def __init__(self, decision_threshold: float = 0.5):
        self.decision_threshold = decision_threshold
    
    def extract_decision_points(self,
                               value_signal: np.ndarray,
                               time_axis: Optional[np.ndarray] = None) -> Dict:
        """
        Extract decision points from value signal.
        
        Args:
            value_signal: Signal representing decision values
            time_axis: Optional time axis
        
        Returns:
            Dictionary with decision points and features
        """
        # Find crossings of threshold
        above_threshold = value_signal > self.decision_threshold
        crossings = np.diff(above_threshold.astype(int))
        decision_indices = np.where(crossings > 0)[0]
        
        if time_axis is not None:
            decision_times = time_axis[decision_indices]
        else:
            decision_times = decision_indices
        
        # Extract features at decision points
        decision_values = value_signal[decision_indices]
        decision_gradients = np.gradient(value_signal)[decision_indices]
        
        return {
            'decision_indices': decision_indices,
            'decision_times': decision_times,
            'decision_values': decision_values,
            'decision_gradients': decision_gradients,
            'n_decisions': len(decision_indices)
        }
    
    def compute_decision_latency(self,
                                 signal: np.ndarray,
                                 decision_times: np.ndarray,
                                 baseline_window: Tuple[float, float] = (-0.5, 0.0)) -> np.ndarray:
        """Compute decision latency from baseline to decision."""
        latencies = []
        
        for decision_time in decision_times:
            baseline_start = decision_time + baseline_window[0]
            baseline_end = decision_time + baseline_window[1]
            
            # Find indices (simplified - assumes time in samples)
            baseline_idx = int(baseline_start)
            decision_idx = int(decision_time)
            
            if baseline_idx >= 0 and decision_idx < len(signal):
                baseline_value = np.mean(signal[baseline_idx:decision_idx])
                latency = decision_time - baseline_idx
                latencies.append(latency)
        
        return np.array(latencies)

