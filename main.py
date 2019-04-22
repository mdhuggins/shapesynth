# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.mixer import *
from common.synth import *
from common.clock import *

from kivy.graphics import Color, Line
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np
import matplotlib.pyplot as plt

from common.kinect import *

from shape import *
from gesture import *

# x, y, and z ranges to define a 3D bounding box
kKinectRange = ( (-500, 500), (-200, 700), (-500, 0) )

# If true, use kinect for gestures. Otherwise, use mouse input.
# Set using sys.argv[1]
USE_KINECT = False

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        self.info = topleft_label()
        self.add_widget(self.info)

        self.audio = Audio(1)
        self.mixer = Mixer()
        self.tempo_map  = SimpleTempoMap(120)
        self.sched = AudioScheduler(self.tempo_map)
        self.sched.set_generator(self.mixer)
        self.audio.set_generator(self.sched)
        Conductor.initialize(self.sched)
        Conductor.start()

        # Set up kinect
        self.kinect = Kinect()
        self.kinect.add_joint(Kinect.kLeftHand)
        self.kinect.add_joint(Kinect.kRightHand)
        self.mouse_pos = None
        self.shape_scale = 500.0 / Window.width # After drawing shapes, transform by this scale factor

        # Set up hold gestures, which trigger shape manipulation
        if USE_KINECT:
            self.gestures = [HoldGesture("create_left", self.get_left_pos, self.on_hold_gesture, lambda x: self.is_in_front(x, 0.05)),
                             HoldGesture("create_right", self.get_right_pos, self.on_hold_gesture, lambda x: self.is_in_front(x, 0.05))]
        else:
            self.gestures = [HoldGesture("create", self.get_mouse_pos, self.on_hold_gesture, None)]

        # Create cursors
        self.margin = np.array([Window.width * 0.05, Window.width * 0.05])
        self.window_size = [Window.width - 2 * self.margin[0], Window.height - 2 * self.margin[1]]
        self.left_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5))
        self.canvas.add(self.left_hand)
        self.right_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5))
        self.canvas.add(self.right_hand)

        self.shapes = []
        self.shape_creator = None

        self.interaction_anims = AnimGroup()
        self.canvas.add(self.interaction_anims)

    def on_update(self) :
        self.info.text = ''

        self.kinect.on_update()
        norm_right = self.get_right_pos(screen=False)
        norm_left = self.get_left_pos(screen=False)
        self.left_hand.set_pos(norm_left)
        self.right_hand.set_pos(norm_right)

        self.audio.on_update()

        for gesture in self.gestures:
            gesture.on_update()

        self.interaction_anims.on_update()

    # Mouse movement callbacks

    def on_touch_down(self, touch):
        self.mouse_pos = np.array(touch.pos)
    def on_touch_up(self, touch):
        self.mouse_pos = None
    def on_touch_move(self, touch):
        self.mouse_pos = np.array(touch.pos)

    # Methods called by gestures to get current positions

    def get_left_pos(self, screen=True):
        pt = self.kinect.get_joint(Kinect.kLeftHand)
        return self.kinect_to_screen(pt) if screen else scale_point(pt, kKinectRange)
    def get_right_pos(self, screen=True):
        pt = self.kinect.get_joint(Kinect.kRightHand)
        return self.kinect_to_screen(pt) if screen else scale_point(pt, kKinectRange)
    def get_mouse_pos(self):
        return self.mouse_pos

    def kinect_to_screen(self, kinect_pt):
        """
        Returns a numpy array representing the location of the given kinect point
        (3D) into screen space (2D). The third dimension value is simply scaled
        from 0 to 1 using the scale_point function.
        """
        scaled = scale_point(kinect_pt, kKinectRange)
        return np.concatenate([scaled[:2] * self.window_size + self.margin, scaled[2:]])

    def is_in_front(self, point, threshold):
        """
        Returns True if the point is outside an ellipsoid that is at a z-value
        of `threshold` for most of the interactive space.
        """
        val = point[0] ** 2 / (Window.width * 1.5) ** 2 + point[1] ** 2 / (Window.height * 1.5) ** 2 + (1.0 - point[2]) ** 2 / (1.0 - threshold) ** 2
        #if val < 1.0:
        #    print(point)
        return val >= 1.0

    # Interaction callbacks

    def on_hold_gesture(self, gesture):
        """Called when a hold gesture is completed."""

        # Initialize the shape gesture using the same point source as this hold gesture gesture
        self.shape_creator = ShapeCreator((0.5, 0.7, 0.8), gesture.source, self.on_shape_creator_complete)
        self.interaction_anims.add(self.shape_creator)

        # Disable other gestures while creating a shape
        for gest in self.gestures:
            gest.set_enabled(False)

    def on_shape_creator_complete(self, points):
        """
        Called when the shape creator detects the shape is closed.
        """
        # Translate and scale the points around the first point
        new_points = [((points[i] - points[0]) * self.shape_scale + points[0], (points[i + 1] - points[1]) * self.shape_scale + points[1]) for i in range(0, len(points), 2)]

        # Create and add the shape
        new_shape = Shape(new_points, (0.5, 0.7, 0.8), self.sched, self.mixer)
        self.shapes.append(new_shape)

        # Animate out shape creator
        def on_creator_completion():
            self.interaction_anims.remove(self.shape_creator)
            self.shape_creator = None
            self.canvas.add(new_shape)
        self.shape_creator.hide_transition([coord for point in new_points for coord in point], on_creator_completion)

        # Reenable other gestures
        for gesture in self.gestures:
            gesture.set_enabled(True)

# pass in which MainWidget to run as a command-line arg
if __name__ == '__main__':
    if len(sys.argv) > 1:
        USE_KINECT = True if sys.argv[1].lower() in ['true', '1'] else False
    print("Using Kinect" if USE_KINECT else "Using mouse-based gestures")
    run(MainWidget, title="ShapeSynth")