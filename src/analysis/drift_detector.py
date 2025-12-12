import numpy as np


class DriftDetector:
    def __init__(self, window_size=10, threshold_std=2.0):
        self.history = []
        self.window_size = window_size
        self.threshold_std = threshold_std

    def update_and_check(self, new_score: float) -> tuple:
        """
        Adds new score to history and checks for Z-Score anomaly.
        Returns: (is_anomaly (bool), drift_score (float))
        """
        is_anomaly = False
        drift_score = 0.0

        if len(self.history) >= 5:
            mean = np.mean(self.history)
            std = np.std(self.history) + 1e-9  # Avoid div by zero

            drift_score = (new_score - mean) / std

            if abs(drift_score) > self.threshold_std:
                is_anomaly = True

        self.history.append(new_score)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        return is_anomaly, drift_score