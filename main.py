# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.mixer import *
from common.synth import *
from common.clock import *
from common.writer import *

from kivy.graphics import Color, Line
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np
import matplotlib.pyplot as plt

from common.kinect import *

from synth.reverb import Reverb

from shape import *
from gesture import *
from keyboard import *
from color import *
from measure_bar import *

# x, y, and z ranges to define a 3D bounding box
kKinectRange = ( (-500, 500), (-200, 700), (-500, 0) )

# If true, use kinect for gestures. Otherwise, use mouse input.
# Set using sys.argv[1]
USE_KINECT = False

HARMONIES = [
    [0, 4, 7, 9],
    [2, 5, 9, 0],
    [4, 7, 11, 0],
    [5, 9, 0, 2],
    [7, 11, 2, 4],
    [9, 0, 4, 7]
]

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        Window.bind(on_request_close=self.on_request_close)

        self.writer = AudioWriter('data') # for debugging audio output
        self.audio = Audio(1, self.writer.add_audio)
        self.mixer = Mixer()
        self.mixer.set_gain(1)
        self.bpm = 92
        self.tempo_map  = SimpleTempoMap(self.bpm)
        self.sched = AudioScheduler(self.tempo_map)
        self.sched.set_generator(self.mixer)

        #master_reverb = Reverb(self.sched)

        self.audio.set_generator(self.sched) #master_reverb)
        SamplerManager.initialize()
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
            self.gestures = [HoldGesture("create", self.get_left_pos, self.on_hold_gesture, self.is_in_front),
                             HoldGesture("create", self.get_right_pos, self.on_hold_gesture, self.is_in_front)]
        else:
            self.gestures = [HoldGesture("create", self.get_mouse_pos, self.on_hold_gesture, None)]

        # Create cursors
        self.margin = np.zeros(2)
        self.window_size = [Window.width - 2 * self.margin[0], Window.height - 2 * self.margin[1]]
        self.left_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5), border=False)
        self.canvas.add(self.left_hand)
        self.right_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5), border=False)
        self.canvas.add(self.right_hand)

        self.palette = ColorPalette()
        self.shapes = []
        self.shape_creator = None
        self.shape_editor = None

        self.interaction_anims = AnimGroup()
        self.canvas.add(self.interaction_anims)

        self.measure_bar = MeasureBar(Window.width, int(Window.height*0.02), self.palette, self.sched)
        self.canvas.add(self.measure_bar)

        # MIDI
        self.keyboard = Keyboard(self.on_chord_change)

        self.label = Label(text = "", valign='top', halign='center', font_size='20sp',
                  pos=(Window.width / 2.0 - 50.0, 50.0), font_name='res/Exo-Bold.otf',
                  text_size=(Window.width, 200.0))
        self.add_widget(self.label)

        # In case the window size changes
        self.last_width = Window.width
        self.last_height = Window.height

    def on_request_close(self, *args):
        Conductor.stop()
        for shape in self.shapes:
            shape.composer.stop()
        SamplerManager.stop_workers()
        return False

    def on_update(self) :
        # Check if window changed size
        if Window.height != self.last_height or Window.width != self.last_width:
            print("Window size changed!")

            self.last_width = Window.width
            self.last_height = Window.height

            # Update components
            self.measure_bar.update_size(Window.width, int(Window.height*0.02))


        self.kinect.on_update()
        norm_right = self.get_right_pos(screen=False)
        norm_left = self.get_left_pos(screen=False)
        self.left_hand.set_pos(norm_left)
        self.right_hand.set_pos(norm_right)

        if USE_KINECT and not self.is_tracking():
            self.label.text = ''

        SamplerManager.on_update()
        self.audio.on_update()

        for gesture in self.gestures:
            gesture.on_update()

        self.measure_bar.on_update()

        self.interaction_anims.on_update()
        if len(self.label.text) == 0:
            if USE_KINECT and not self.is_tracking():
                self.label.text = 'Hold your arms up to begin.'
            elif len(self.shapes) == 0:
                self.label.text = 'Extend your hand to start drawing a shape.'
            else:
                self.label.text = 'harmony: ' + Conductor.harmony_string()

    # Change harmony with keystrokes

    def on_key_down(self, keycode, modifiers):
        harmony = lookup(keycode[1], '123456', HARMONIES)
        if harmony is not None:
            is_different = (harmony != Conductor.harmony)
            Conductor.harmony = harmony
            if is_different:
                self.measure_bar.update_color()
                for shape in self.shapes:
                    shape.composer.clear_notes()
                self.label.text = ''

        if keycode[1] == 'spacebar':
            for shape in self.shapes:
                shape.composer.toggle()

        if keycode[1] == 'z':
            self.writer.toggle()

    def on_chord_change(self, pitches):
        """
        Called when the keyboard's played pitches change.
        """
        new_harmony = [pitch % 12 for pitch in sorted(pitches)]
        if new_harmony != Conductor.harmony:
            Conductor.harmony = new_harmony
            self.measure_bar.update_color()
            for shape in self.shapes:
                shape.composer.clear_notes()
        self.label.text = ''

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

    def is_tracking(self):
        """Returns whether or not the Kinect is currently tracking a user."""
        return sum(self.kinect.get_joint(Kinect.kLeftHand)) != 0

    def kinect_to_screen(self, kinect_pt):
        """
        Returns a numpy array representing the location of the given kinect point
        (3D) into screen space (2D). The third dimension value is simply scaled
        from 0 to 1 using the scale_point function.
        """
        scaled = scale_point(kinect_pt, kKinectRange)
        return np.concatenate([scaled[:2] * self.window_size + self.margin, scaled[2:]])

    def is_in_front(self, point, threshold=0.3):
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

        if gesture.identifier == "create":
            if any(g for g in self.gestures if g.identifier != "create" and g.is_recognizing()):
                print("Another gesture is recognizing")
                return
            # Initialize the shape gesture using the same point source as this hold gesture gesture
            self.shape_creator = ShapeCreator((0.5, 0.7, 0.8), gesture.source, self.on_shape_creator_complete)
            self.interaction_anims.add(self.shape_creator)
            self.label.text = "Move your hand to draw a closed shape."

            # Disable other gestures while creating a shape
            for gest in self.gestures:
                gest.set_enabled(False)

        elif gesture.identifier in self.shapes:
            editing_shape = gesture.identifier
            self.interaction_anims.remove(editing_shape)
            self.shape_editor = ShapeEditor((0.4, 0.7, 0.8), editing_shape, gesture.source, self.on_shape_editor_complete, end=ShapeEditor.END_POSE if USE_KINECT else ShapeEditor.END_CLICK)
            self.interaction_anims.add(self.shape_editor)
            self.label.text = "Move your hand to alter the shape position and size."

            # Disable other gestures while editing a shape
            for gest in self.gestures:
                gest.set_enabled(False)

    def on_shape_creator_complete(self, points):
        """
        Called when the shape creator detects the shape is closed.
        """
        if len(points) > 0:
            # Translate and scale the points around the first point
            new_points = [((points[i] - points[0]) * self.shape_scale + points[0], (points[i + 1] - points[1]) * self.shape_scale + points[1]) for i in range(0, len(points), 2)]

            # Create and add the shape
            new_shape = Shape(new_points, self.palette, self.sched, self.mixer)
            self.shapes.append(new_shape)

            # Animate out shape creator
            def on_creator_completion():
                self.interaction_anims.remove(self.shape_creator)
                self.shape_creator = None
                self.interaction_anims.add(new_shape)
            self.shape_creator.hide_transition([coord for point in new_points for coord in point], on_creator_completion)

            # Add hold gestures for this shape
            if USE_KINECT:
                self.gestures.insert(0, HoldGesture(new_shape, self.get_left_pos, self.on_hold_gesture, lambda x: self.is_in_front(x) and new_shape.hit_test(x)))
                self.gestures.insert(0, HoldGesture(new_shape, self.get_right_pos, self.on_hold_gesture, lambda x: self.is_in_front(x) and new_shape.hit_test(x)))
            else:
                self.gestures.insert(0, HoldGesture(new_shape, self.get_mouse_pos, self.on_hold_gesture, hit_test=new_shape.hit_test))
        else:
            def on_creator_completion():
                self.interaction_anims.remove(self.shape_creator)
                self.shape_creator = None
            self.shape_creator.hide_transition([], on_creator_completion)

        # Reenable other gestures
        for gesture in self.gestures:
            gesture.set_enabled(True)

        self.label.text = ''

    def on_shape_editor_complete(self, editor, removed):
        """
        Called when the shape editor detects the user is finished.
        """
        def on_editor_completion():
            self.interaction_anims.remove(self.shape_editor)
            self.shape_editor = None
        self.shape_editor.hide_transition(on_editor_completion, removed)
        if removed:
            self.shapes.remove(editor.shape)
            editor.shape.composer.stop()
        else:
            editor.shape.update_sound()
            self.interaction_anims.add(editor.shape)

        # Reenable other gestures
        for gesture in self.gestures:
            gesture.set_enabled(True)

        self.label.text = ''

# pass in which MainWidget to run as a command-front_line arg
if __name__ == '__main__':
    if len(sys.argv) > 1:
        USE_KINECT = True if sys.argv[1].lower() in ['true', '1'] else False
    print("Using Kinect" if USE_KINECT else "Using mouse-based gestures")
    run(MainWidget, title="ShapeSynth")
