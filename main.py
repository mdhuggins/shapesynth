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

from kivy.graphics import Color, Line, Rectangle
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock
from kivy.animation import Animation
from kivy.uix.widget import Widget

import numpy as np
import matplotlib.pyplot as plt

from common.kinect import *

from synth.reverb import Reverb

from shape import *
from gesture import *
from keyboard import *
from color import *
from measure_bar import *
from cursor import AnimatedCursor
from grid import *
from background import *

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

        self.palette = ColorPalette()

        # Set up kinect
        self.kinect = Kinect()
        self.kinect.add_joint(Kinect.kLeftHand)
        self.kinect.add_joint(Kinect.kRightHand)
        self.touch_pos = None
        self.mouse_pos = None
        self.shape_scale = 500.0 / Window.width # After drawing shapes, transform by this scale factor

        self.backgrounds = [CloudBackground(3, self.palette, size_range=(600, 1200), alpha_range=(0.7, 1.0))]
        for bg in self.backgrounds:
            self.add_widget(bg)

        # Views
        self.game = Widget()
        self.add_widget(self.game)
        self.splash = Widget()
        self.add_widget(self.splash)

        self.interaction_anims = AnimGroup()
        self.game.canvas.add(self.interaction_anims)

        # Set up hold gestures, cursors, and grid
        self.cursors = AnimGroup()
        self.game.canvas.add(self.cursors)
        self.cursor_map = {}
        self.normal_hsv = (0.55, 0.7, 0.7)
        self.drawing_hsv = (0.5, 0.85, 0.98)
        cursor_kwargs = {} #{"normal_hsv": self.normal_hsv, "drawing_hsv": self.drawing_hsv}

        self.grid = Grid()
        self.cursors.add(self.grid)

        if USE_KINECT:
            self.gestures = [HoldGesture("create", self.get_left_pos, self.on_hold_gesture, self.is_in_front, on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel),
                             HoldGesture("create", self.get_right_pos, self.on_hold_gesture, self.is_in_front, on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel)]
            self.left_hand = AnimatedCursor(self.get_left_pos, 180.0, **cursor_kwargs)
            self.right_hand = AnimatedCursor(self.get_right_pos, 180.0, **cursor_kwargs)
            self.cursors.add(self.left_hand)
            self.cursors.add(self.right_hand)
            self.cursor_map[self.get_left_pos] = self.left_hand
            self.cursor_map[self.get_right_pos] = self.right_hand
        else:
            self.gestures = [HoldGesture("create", self.get_touch_pos, self.on_hold_gesture, None, on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel)]
            self.cursor = AnimatedCursor(self.get_mouse_pos, 120.0, **cursor_kwargs)
            self.cursors.add(self.cursor)
            Window.bind(mouse_pos=self.on_mouse_pos)
            self.cursor_map[self.get_touch_pos] = self.cursor

        self.shapes = []
        self.shape_creator = None
        self.shape_editor = None

        self.measure_bar = MeasureBar(Window.width, int(Window.height*0.02), self.palette, self.sched)
        self.game.canvas.add(self.measure_bar)

        # MIDI
        print("Using port {} for MIDI keyboard".format(KEYBOARD_PORT))
        self.keyboard = Keyboard(self.on_chord_change, port=KEYBOARD_PORT)

        self.label = Label(text = "", valign='top', halign='center', font_size='20sp',
                  pos=(Window.width / 2.0 - 50.0, 50.0), font_name='res/Exo-Bold.otf',
                  text_size=(Window.width, 200.0))
        self.game.add_widget(self.label)

        # Splash
        self.splash_title = Label(text="ShapeSynth", valign='center', halign='center',
                                  font_size='85sp',
                                  pos=(Window.width / 2.0-50, Window.height / 2.0-50),
                                  font_name='res/Exo-Bold.otf',
                                  text_size=(Window.width, Window.height))
        self.splash.add_widget(self.splash_title)

        # Splash Animation
        self.splash.canvas.opacity = 0

        # Fade out
        def hide_label(w): self.remove_widget(w)
        splash_anim1 = Animation(opacity=0, duration=1.5)
        splash_anim1.on_complete = hide_label

        # Hold
        def start_fade(w): splash_anim1.start(w)
        splash_anim0 = Animation(opacity=1, duration=4)
        splash_anim0.on_complete = start_fade

        # Fade in
        def start_hold(w): splash_anim0.start(w)
        splash_anim = Animation(opacity=1, duration=0.25)
        splash_anim.on_complete = start_hold

        splash_anim.start(self.splash.canvas)


        # Game canvas fade in animation
        self.game.canvas.opacity = 0

        canvas_anim1 = Animation(opacity=1, duration=1.5)

        def start_fade_in(w): canvas_anim1.start(w)
        canvas_anim0 = Animation(opacity=0, duration=4.75)
        canvas_anim0.on_complete = start_fade_in

        canvas_anim0.start(self.game.canvas)


        # In case the window size changes
        self.last_width = Window.width
        self.last_height = Window.height

    def on_request_close(self, *args, **kwargs):
        Conductor.stop()
        for shape in self.shapes:
            shape.composer.stop()
        SamplerManager.stop_workers()
        return False

    def on_update(self) :
        # Check if window changed size
        if Window.height != self.last_height or Window.width != self.last_width:
            print("Window size changed!")

            # Update components
            for shape in self.shapes:
                shape.update_for_window_size((self.last_width, self.last_height))

            self.grid.redraw_grid()
            self.measure_bar.update_size(Window.width, int(Window.height*0.02))
            self.splash_title.pos = (Window.width / 2.0-50, Window.height / 2.0-50)
            self.label.pos = (Window.width / 2.0 - 50.0, 50.0)

            self.last_width = Window.width
            self.last_height = Window.height


        self.kinect.on_update()

        if USE_KINECT and not self.is_tracking():
            self.label.text = ''

        SamplerManager.on_update()
        self.audio.on_update()

        for gesture in self.gestures:
            gesture.on_update()

        for bg in self.backgrounds:
            bg.on_update()

        self.measure_bar.on_update()
        self.interaction_anims.on_update()
        self.cursors.on_update()
        if len(self.label.text) == 0:
            if USE_KINECT and not self.is_tracking():
                self.label.text = 'Hold your arms up to begin.'
            elif len(self.shapes) == 0:
                self.label.text = 'Extend your hand to start drawing a shape.'
            else:
                self.label.text = 'Harmony: ' + Conductor.harmony_string()

    # Change harmony with keystrokes

    def on_key_down(self, keycode, modifiers):
        harmony = lookup(keycode[1], '123456', HARMONIES)
        if harmony is not None:
            is_different = (harmony != Conductor.harmony)
            Conductor.set_harmony(harmony)
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
            Conductor.set_harmony(new_harmony)
            self.measure_bar.update_color()
            for shape in self.shapes:
                shape.composer.clear_notes()
        self.label.text = ''

    # Mouse movement callbacks

    def on_touch_down(self, touch):
        self.touch_pos = np.array(touch.pos)
    def on_touch_up(self, touch):
        self.touch_pos = None
    def on_touch_move(self, touch):
        self.touch_pos = np.array(touch.pos)
    def on_mouse_pos(self, window, pos):
        self.mouse_pos = np.array(pos)

    # Methods called by gestures to get current positions

    def get_left_pos(self, screen=True):
        pt = self.kinect.get_joint(Kinect.kLeftHand)
        return self.kinect_to_screen(pt) if screen else scale_point(pt, kKinectRange)
    def get_right_pos(self, screen=True):
        pt = self.kinect.get_joint(Kinect.kRightHand)
        return self.kinect_to_screen(pt) if screen else scale_point(pt, kKinectRange)
    def get_touch_pos(self):
        return self.touch_pos
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
        return np.concatenate([scaled[:2] * np.array([Window.width, Window.height]), scaled[2:]])

    def is_in_front(self, point, threshold=0.3):
        """
        Returns True if the point is outside an ellipsoid that is at a z-value
        of `threshold` for most of the interactive space.
        """
        if point[1] / Window.height <= 0.1:
            return False

        val = point[0] ** 2 / (Window.width * 1.5) ** 2 + point[1] ** 2 / (Window.height * 1.5) ** 2 + (1.0 - point[2]) ** 2 / (1.0 - threshold) ** 2
        #if val < 1.0:
        #    print(point)
        return val >= 1.0

    # Interaction callbacks

    def on_hold_gesture(self, gesture):
        """Called when a hold gesture is completed."""

        if gesture.identifier == "create":
            cursor = self.cursor_map[gesture.source]
            cursor.set_state(AnimatedCursor.DRAWING)

            if any(g for g in self.gestures if g.identifier != "create" and g.is_recognizing()):
                print("Another gesture is recognizing")
                return
            # Initialize the shape gesture using the same point source as this hold gesture gesture
            self.shape_creator = ShapeCreator(self.drawing_hsv, gesture.source, self.on_shape_creator_complete)
            self.interaction_anims.add(self.shape_creator)
            self.label.text = "Move your hand to draw a closed shape."

            # Disable other gestures while creating a shape
            for gest in self.gestures:
                gest.set_enabled(False)

        elif gesture.identifier in self.shapes:
            editing_shape = gesture.identifier

            cursor = self.cursor_map[gesture.source]
            cursor.editing_hsv = gesture.identifier.hsv
            cursor.set_state(AnimatedCursor.EDITING)

            self.interaction_anims.remove(editing_shape)
            self.shape_editor = ShapeEditor(gesture.identifier.hsv, editing_shape, gesture.source, self.on_shape_editor_complete, end=ShapeEditor.END_POSE if USE_KINECT else ShapeEditor.END_CLICK)
            self.interaction_anims.add(self.shape_editor)
            self.label.text = "Move your hand to change the shape's position and size."

            # Disable other gestures while editing a shape
            for gest in self.gestures:
                gest.set_enabled(False)

    def on_hold_gesture_trigger(self, gesture):
        """Called when a hold gesture begins."""
        cursor = self.cursor_map[gesture.source]
        self.grid.set_grid_visible(True)
        if gesture.identifier == "create":
            self.grid.make_target_animation(gesture.original_pos, gesture.hold_time)
        cursor.set_state(AnimatedCursor.HOLDING)
        for other_gesture in self.gestures:
            if other_gesture != gesture:
                other_gesture.set_enabled(False)

    def on_hold_gesture_cancel(self, gesture):
        """Called when a hold gesture is canceled."""
        cursor = self.cursor_map[gesture.source]
        cursor.set_state(AnimatedCursor.NORMAL)
        self.grid.set_grid_visible(False)
        for other_gesture in self.gestures:
            other_gesture.set_enabled(True)


    def on_shape_creator_complete(self, points):
        """
        Called when the shape creator detects the shape is closed.
        """
        if self.shape_creator is None:
            return
        cursor = self.cursor_map[self.shape_creator.source]
        cursor.set_state(AnimatedCursor.NORMAL)
        self.grid.set_grid_visible(False)

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
                self.gestures.insert(0, HoldGesture(new_shape, self.get_left_pos, self.on_hold_gesture, lambda x: self.is_in_front(x) and new_shape.hit_test(x), on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel))
                self.gestures.insert(0, HoldGesture(new_shape, self.get_right_pos, self.on_hold_gesture, lambda x: self.is_in_front(x) and new_shape.hit_test(x), on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel))
            else:
                self.gestures.insert(0, HoldGesture(new_shape, self.get_touch_pos, self.on_hold_gesture, hit_test=new_shape.hit_test, on_trigger=self.on_hold_gesture_trigger, on_cancel=self.on_hold_gesture_cancel))
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
        if self.shape_editor is None:
            return
        cursor = self.cursor_map[self.shape_editor.source]
        cursor.set_state(AnimatedCursor.NORMAL)
        self.grid.set_grid_visible(False)

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

    KEYBOARD_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print("Using Kinect" if USE_KINECT else "Using mouse-based gestures")
    run(MainWidget, title="ShapeSynth")
