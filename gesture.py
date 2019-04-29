import time
import numpy as np

class Gesture(object):
    """
    Abstract base class for gestures that receive current positions of the mouse
    or hand, and call the appropriate callback when the gesture is recognized.
    """

    def __init__(self, identifier, source, callback):
        """
        identifier: an object that clients can use to identify what this gesture
            triggers
        source: a function that returns a current position or None; called in the
            update loop
        callback: a function to be called when the gesture is fired, with one
            parameter for the gesture object
        """
        super(Gesture, self).__init__()
        self.identifier = identifier
        self.callback = callback
        self.source = source
        self.enabled = True

    def on_update(self):
        """
        Updates the gesture by calling the gesture's source to retrieve a current
        position. Calls the callback if the gesture is triggered.
        """
        pass

    def set_enabled(self, enabled):
        self.enabled = enabled

class HoldGesture(Gesture):
    """
    Recognizes a gesture in which the hand position remains in the same area for
    a fixed period of time.
    """

    def __init__(self, identifier, source, callback, hit_test=None, hold_time=0.75):
        """
        Creates a hold gesture. hit_test may be a function taking a point and
        returning a boolean indicating whether the point is within the region
        of interest for this hold gesture.
        """
        super(HoldGesture, self).__init__(identifier, source, callback)
        self.hit_test = hit_test
        self.start_time = None
        self.original_pos = None
        self.hold_time = hold_time

    def on_update(self):
        if not self.enabled: return

        pos = self.source()

        # Reset if no position available, or not in desired window
        if pos is None or (self.hit_test is not None and not self.hit_test(pos)):
            self.original_pos = None
            self.start_time = None
            return

        if self.original_pos is None or self.start_time is None:
            self.original_pos = pos
            self.start_time = time.time()
            return

        # Check that position didn't move for self.hold_time
        delta = np.linalg.norm(self.original_pos - pos)
        if delta < 16.0:
            if time.time() - self.start_time >= self.hold_time:
                self.callback(self)
                self.start_time = None
                self.original_pos = None
        else:
            self.original_pos = None
            self.start_time = None

    def set_enabled(self, enabled):
        super(HoldGesture, self).set_enabled(enabled)
        if not enabled:
            self.original_pos = None
            self.start_time = None
