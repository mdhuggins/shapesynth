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

    def is_recognizing(self):
        """Returns whether the gesture is currently accepting a potential gesture."""
        return False

class HoldGesture(Gesture):
    """
    Recognizes a gesture in which the hand position remains in the same area for
    a fixed period of time.
    """

    def __init__(self, identifier, source, callback, hit_test=None, hold_time=1.0, on_trigger=None, on_cancel=None):
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
        self.recognizing = False
        self.on_trigger = on_trigger
        self.on_cancel = on_cancel

    def on_update(self):
        if not self.enabled: return

        pos = self.source()

        # Reset if no position available, or not in desired window
        if pos is None or (self.hit_test is not None and not self.hit_test(pos)):
            if self.recognizing:
                self.on_cancel(self)
            self.original_pos = None
            self.start_time = None
            self.recognizing = False
            return

        if self.original_pos is None or self.start_time is None:
            self.original_pos = pos
            self.start_time = time.time()
            return

        # Check that position didn't move for self.hold_time
        delta = np.linalg.norm(self.original_pos - pos)
        if delta < 16.0:
            self.recognizing = True
            if time.time() - self.start_time >= self.hold_time:
                self.callback(self)
                self.start_time = None
                self.original_pos = None
                self.recognizing = False
            elif time.time() - self.start_time >= self.hold_time / 4.0 and self.on_trigger is not None:
                self.on_trigger(self)
        else:
            if self.recognizing:
                self.on_cancel(self)
            self.recognizing = False
            self.original_pos = None
            self.start_time = None

    def set_enabled(self, enabled):
        super(HoldGesture, self).set_enabled(enabled)
        if not enabled:
            self.original_pos = None
            self.start_time = None

    def is_recognizing(self):
        return self.recognizing
