from common.core import *
from common.gfxutil import *
from common.audio import *
from common.mixer import *
from common.synth import *
from common.clock import *

from kivy.core.image import Image
from kivy.graphics import Color, Line, Mesh, Translate, Scale, BindTexture, Rotate
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import lfilter, butter
import tripy
import matplotlib.pyplot as plt

from composer import *
from synth.shape_synth import *
from gesture import HoldGesture
from color import ColorPalette

class Shape(InstructionGroup):
    """
    Represents a 2D shape on the canvas.
    """

    def __init__(self, points, palette, sched, mixer):
        """
        points - a list of points bounding the shape
        palette - a ColorPalette object that returns colors to use
        sched - a scheduler on which to build the composer
        mixer - the mixer to which to add notes
        """
        super(Shape, self).__init__()
        self.points = points
        self.palette = palette
        self.hsv = (0, 1, 1)
        self.shadow_index = len(self.children)
        self.color_frozen = False
        self.color_anim = None
        self.time = 0.0

        self.make_shape_properties()

        self.rotation = Rotate(0.0)
        self.rotation_rate = np.random.uniform(-15.0, 15.0)
        self.add(PushMatrix())
        self.translate = Translate(*(self.center * np.array([Window.width, Window.height])))
        self.add(self.translate)
        self.add(self.rotation)
        self.scale = Scale(1.0, 1.0)
        self.add(self.scale)
        self.back_translate = Translate(*(-self.center * np.array([Window.width, Window.height])))
        self.add(self.back_translate)
        self.fill_color = Color(hsv=self.hsv)
        self.fill_color.a = 0.5
        self.add(self.fill_color)
        self.mesh = Mesh(mode='triangles')
        self.update_mesh()
        self.add(self.mesh)
        self.stroke_color = Color(hsv=self.hsv)
        self.add(self.stroke_color)
        self.curve = Line(points=[coord for point in self.points for coord in point], segments=20 * len(self.points), loop=True)
        self.curve.width = 3.0
        self.add(self.curve)
        self.add(PopMatrix())

        self.make_synth()
        self.make_composer(sched, mixer)

        self.colors = [self.fill_color, self.stroke_color]
        self.shadow_anims = {}
        self.shadow_reenable_time = 0
        self.update_color(animated=False)

    def set_points(self, points):
        """Takes the given numpy nx2 array of points and sets the shape's points
        accordingly."""
        self.points = [tuple(row) for row in points]
        self.curve.points = [coord for point in self.points for coord in point]
        self.update_mesh()
        self.make_shape_properties()
        self.translate.xy = (self.center * np.array([Window.width, Window.height]))
        self.back_translate.xy = (-self.center * np.array([Window.width, Window.height]))

        for (shadow, color) in self.shadow_anims:
            shadow.pos = self.center * np.array([Window.width, Window.height]) - np.array(shadow.size) / 2.0

    def set_alpha(self, alpha):
        self.fill_color.a = alpha / 2.0
        self.stroke_color.a = alpha

    def hit_test(self, point):
        """
        Returns True if the given point is contained within the mesh.
        """
        return point[0] >= self.min_x and point[0] <= self.max_x and point[1] >= self.min_y and point[1] <= self.max_y

    def update_mesh(self):
        """
        Sets the Mesh's vertices and indices to display this shape.
        """
        vertices = []
        theta = 2 * np.pi / len(self.points)
        for point in self.points:
            vertices += [point[0], point[1], 0, 0]

        # Use earclipping algorithm to triangulate the shape
        triangles = tripy.earclip(self.points)
        indices = []
        for point1, point2, point3 in triangles:
            indices += [self.points.index(point1), self.points.index(point2), self.points.index(point3)]

        self.mesh.vertices = vertices
        self.mesh.indices = indices

    def update_sound(self):
        """
        Refreshes the sonic properties of the shape.
        """
        was_on = self.composer.playing
        self.composer.stop()
        self.make_shape_properties()
        self.make_synth()
        self.make_composer(self.composer.sched, self.composer.mixer, was_on)

    def make_shape_properties(self):
        """
        Computes properties about this shape that can be used to make a synth
        and/or composer.
        """
        point_array = np.array(self.points)
        self.center = np.mean(point_array, axis=0) / np.array([Window.width, Window.height])
        self.min_x = np.min(point_array[:,0])
        self.max_x = np.max(point_array[:,0])
        self.min_y = np.min(point_array[:,1])
        self.max_y = np.max(point_array[:,1])
        self.area = (self.max_x - self.min_x) * (self.max_y - self.min_y)

        self.roughness = 1  # In range [0,1]

        r = 0

        for i in range(len(self.points)-4):
            p0, p1, p2 = self.points[i:i+5:2]

            v0 = np.subtract(p1,p0)
            v1 = np.subtract(p2,p1)

            # In range [-1,1]
            dot = np.dot(v0, v1) / (np.linalg.norm(v0) + np.linalg.norm(v1))

            # In range [0,1]
            dot = (dot+1)/2

            r += 1-np.tanh(dot)


        # r /= len(self.points)

        self.roughness = np.clip(r / 10.0, 0.0, 1.0)
        print(self.roughness)

    def make_synth(self):
        """
        Creates a ShapeSynth for this shape. TODO: Use properties other than
        center and area.
        """
        min_gain = 0.05
        max_gain = 0.6
        gain = np.clip(self.area / 6000.0 * (max_gain - min_gain) + min_gain, min_gain, max_gain)

        self.synth = ShapeSynth(self.center[0], self.center[1], gain, self.roughness)
        self.synth.on_note = self.on_note

    def make_composer(self, sched, mixer, start=True):
        """
        Builds this shape's Composer using its location and properties of its
        vertices.
        """
        self.composer = Composer(sched, mixer, self.synth.make_note)
        self.composer.pitch_level = np.sqrt(self.center[0] * 0.7)
        self.composer.pitch_variance = (self.center[0] / 2.0) ** 2
        self.composer.complexity = self.roughness
        self.composer.harmonic_obedience = np.sqrt(1.0 - self.roughness)
        self.composer.bass_preference = 1 - self.center[0]
        self.composer.arpeggio_preference = 2.0 * self.center[0] * (1 - self.center[0])
        self.composer.update_interval = 4 if self.center[0] > 0.3 else 8
        self.composer.velocity_level = 0.5
        self.composer.velocity_variance = self.center[0] * (1 - self.center[0])
        self.composer.update_callback = self.update_color

        if start:
            self.composer.start()

    def on_note(self, pitch, velocity, dur):
        """Called when the ShapeSynth plays a note."""
        if self.color_frozen: return
        if self.time < self.shadow_reenable_time: return

        center = ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
        dim = max(self.max_x - self.min_x, self.max_y - self.min_y) * 2.5

        new_circle = Rectangle(pos=(center[0] - dim / 2, center[1] - dim / 2), size=(dim, dim))
        #new_circle = CEllipse(cpos=center, csize=(dim, dim), texture=tex)
        color = Color(hsv=self.fill_color.hsv)
        color.a = 0.3
        self.colors.append(color)
        duration = 4.0
        self.shadow_anims[(new_circle, color)] = (self.time, KFAnim((0.0, dim), (duration, dim * 10.0)), KFAnim((0.0, 0.3), (duration, 0.0)))
        self.insert(self.shadow_index, new_circle)
        self.insert(self.shadow_index, color)
        self.insert(self.shadow_index, BindTexture(source='res/blur_circle.png'))
        self.shadow_reenable_time = self.time + 1.0

    def on_update(self, dt):
        if self.color_anim is not None:
            start_time, anim = self.color_anim
            new_color = anim.eval(self.time - start_time)
            for color in self.colors:
                old_a = color.a
                color.hsv = new_color
                color.a = old_a
            if not anim.is_active(self.time - start_time):
                self.color_anim = None

        kill_set = set()
        for (circle, color), (start_time, size_anim, alpha_anim) in self.shadow_anims.items():
            new_dim = size_anim.eval(self.time - start_time)
            circle.pos = (circle.pos[0] + circle.size[0] / 2.0 - new_dim / 2.0, circle.pos[1] + circle.size[1] / 2.0 - new_dim / 2.0)
            circle.size = (new_dim, new_dim)
            color.a = alpha_anim.eval(self.time - start_time)
            if not alpha_anim.is_active(self.time - start_time):
                kill_set.add((circle, color))
                self.remove(circle)
                self.remove(color)
                self.colors.remove(color)
        self.shadow_anims = {k: v for k, v in self.shadow_anims.items() if k not in kill_set}

        self.rotation.angle += dt * self.rotation_rate * len(self.shadow_anims)
        if self.composer.playing:
            new_scale = np.sin(self.time * np.pi / 2.0) * 0.15 + 1.0
            self.scale.x = new_scale
            self.scale.y = new_scale
        self.time += dt

    def set_color_frozen(self, flag):
        """Controls whether the shape color is allowed to change."""
        self.color_frozen = flag

    def update_color(self, animated=True):
        """Called when the composer generates new notes."""
        if self.color_frozen: return
        if self.color_anim is not None: return

        new_hsv = self.palette.new_color(self.composer.pitch_level)
        self.color_anim = (self.time, KFAnim((0.0, *self.hsv), (1.0 if animated else 0.0, *new_hsv)))
        self.hsv = new_hsv

SHAPE_CLOSE_THRESHOLD = 40
MAX_DISTANCE_THRESHOLD = 80

def spring_timing_function(duration):
    times = np.array([0.0, 0.7, 0.85, 1.0]) * duration
    values = np.array([0.0, 1.05, 0.9, 1.0])
    def timing(t):
        if t >= duration:
            return None
        return np.interp(t, times, values)
    return timing

class ShapeCreator(InstructionGroup):
    """
    Provides a UI for creating a shape using Kinect gestures.
    """
    def __init__(self, hsv, source, on_complete):
        super(ShapeCreator, self).__init__()
        self.source = source

        self.points = []
        self.hsv = hsv
        self.bg_color = Color(hsv=self.hsv)
        self.bg_color.a = 0.0
        self.bg_anim = KFAnim((0.0, 0.0), (0.5, 0.3))
        self.shape_alpha_anim = None
        self.add(self.bg_color)
        self.add(Rectangle(pos=(0,0), size=(Window.width, Window.height)))

        self.color = Color(hsv=self.hsv)
        self.add(self.color)
        self.line = Line(points=self.points)
        self.line.width = 6.0
        self.add(self.line)

        self.accepting_points = True
        self.gesture_pos_idx = 0
        self.on_complete = on_complete
        self.max_distance = 0.0
        self.time = 0.0
        self.shape_anim_points = None
        self.shape_timing = None
        self.anim_completion = None

    def add_position(self, pos):
        """
        Updates the shape using the given hand position.
        """
        if len(self.points) > 2:
            last_dist = np.linalg.norm(np.array(pos) - np.array(self.points[-2:]))
            if last_dist < 5:
                return

        self.points += pos
        self.line.points = self.points

        # Check if the last point is close enough to the first point
        # while also having gone sufficiently far away
        if len(self.points) > 2:
            dist = np.linalg.norm(np.array(pos) - np.array(self.points[:2]))
            if dist < SHAPE_CLOSE_THRESHOLD and self.max_distance >= MAX_DISTANCE_THRESHOLD:
                self.line.points = self.points
                self.accepting_points = False
                self.smooth_shape()
                self.on_complete(self.points)
            elif dist < SHAPE_CLOSE_THRESHOLD and self.max_distance <= MAX_DISTANCE_THRESHOLD and len(self.points) > 15:
                self.accepting_points = False
                self.on_complete([])
            elif dist > self.max_distance:
                self.max_distance = dist

    def smooth_shape(self):
        """Updates the points to be smoother and form a closed boundary."""
        # Apply LPF to the points to smooth out the shape
        points = np.array([(self.points[i], self.points[i + 1]) for i in range(0, len(self.points), 2)])
        filter = butter(4, 0.2, btype='lowpass')
        point_x = lfilter(*filter, np.concatenate([points[::-1,0], points[:,0]]))[len(points):]
        point_y = lfilter(*filter, np.concatenate([points[::-1,1], points[:,1]]))[len(points):]
        new_points = [coord for i in range(len(point_x)) for coord in [point_x[i], point_y[i]]]

        # Make sure the shape is closed
        new_points += (point_x[0], point_y[0])

        self.points = new_points
        self.line.points = self.points

    def on_update(self, dt):
        # Handle the background animation
        if self.bg_anim is not None:
            self.bg_color.a = self.bg_anim.eval(self.time)
            if not self.bg_anim.is_active(self.time):
                self.bg_anim = None
                if self.anim_completion is not None:
                    self.anim_completion()

        # Handle the shape animation (with spring function)
        if self.shape_anim_points is not None:
            old, new = self.shape_anim_points
            t = self.shape_timing(self.time)
            if t is None:
                self.shape_anim_points = None
                self.shape_timing = None
            else:
                self.line.points = (old * (1 - t) + new * t).tolist()

        if self.shape_alpha_anim is not None:
            self.color.a = self.shape_alpha_anim.eval(self.time)
            if not self.shape_alpha_anim.is_active(self.time):
                self.shape_alpha_anim = None

        self.time += dt

        # Handle new points
        if self.accepting_points:
            new_pos = self.source()
            if (len(new_pos) > 2 and new_pos[2] > 0.7) or len(self.points) > 800:
                # The user stopped drawing - cancel
                self.accepting_points = False
                self.on_complete([])
                return
            # Only accept every third point (since data can be noisy)
            self.gesture_pos_idx = (self.gesture_pos_idx + 1) % 3
            if self.gesture_pos_idx == 0 and new_pos is not None:
                self.add_position(new_pos[:2].tolist())

    def hide_transition(self, final_points, callback):
        """
        Performs a hide animation while interpolating the drawn line boundary to
        the given set of points. Calls the given callback function on completion.
        """
        self.time = 0.0
        self.bg_anim = KFAnim((0.0, self.bg_color.a), (1.0, 0.0))
        if len(final_points) > 0:
            self.shape_anim_points = (np.array(self.points), np.array(final_points))
            self.shape_timing = spring_timing_function(1.0)
        else:
            self.shape_alpha_anim = KFAnim((0.0, 1.0), (1.0, 0.0))
        self.anim_completion = callback

class ShapeEditor(InstructionGroup):
    """
    Provides a UI for translating, scaling (TODO: distorting) shapes.
    """

    END_CLICK = "click"
    END_POSE = "pose"

    def __init__(self, hsv, shape, source, on_complete, end="click"):
        """
        hsv: tuple (hue, saturation, value) for the background color
        shape: the shape to edit
        source: a callable that generates touch points
        on_complete: function called when the edit is complete, with two parameters:
            the shape editor, and whether the shape was removed
        end: the gesture type that ends editing
        """
        super(ShapeEditor, self).__init__()
        self.source = source
        self.shape = shape

        self.hsv = hsv
        self.bg_color = Color(hsv=self.hsv)
        self.bg_color.a = 0.0
        self.bg_anim = KFAnim((0.0, 0.0), (0.5, 0.3))
        self.add(self.bg_color)
        self.add(Rectangle(pos=(0,0), size=(Window.width, Window.height)))
        self.shape_center = self.shape.center * np.array([Window.width, Window.height])

        self.add(PushMatrix())
        self.translate = Translate(0, 0)
        self.add(self.translate)
        #self.back_translate = Translate(x=0.0, y=0.0)
        #self.add(self.back_translate)
        self.scale = Scale(1.0)
        self.scale.origin = self.shape_center
        self.add(self.scale)
        self.add(self.shape)
        self.add(PopMatrix())

        self.accepting_points = True
        self.gesture_pos_idx = 0
        self.on_complete = on_complete
        self.time = 0.0
        self.scale_anim = None
        self.alpha_anim = None
        self.anim_completion = None
        self.end_mode = end
        self.original_position = None
        self.current_position = None
        self.last_position = None
        if self.end_mode == ShapeEditor.END_POSE:
            self.end_hold = HoldGesture(None, self.get_current_pos, self.end_gesture)
            self.end_hold.set_enabled(False)
            def enable_gesture(ignore):
                self.end_hold.set_enabled(True)
            kivyClock.schedule_once(enable_gesture, 1.0)

    def add_position(self, pos):
        """
        Updates the shape using the given hand position (in 3D if available).
        """
        self.last_position = self.current_position
        self.current_position = pos
        if self.original_position is None:
            self.original_position = pos
            return

        self.translate.xy = pos[:2] - self.original_position[:2]
        if len(pos) > 2:
            new_scale = 1 + (pos[2] - self.original_position[2])
            if pos[2] > 0.8 and self.last_position is not None and (pos[2] - self.last_position[2]) > 0.2:
                self.accepting_points = False
                self.on_complete(self, True)
                return
            self.scale.x = new_scale
            self.scale.y = new_scale

    def get_current_pos(self):
        return self.current_position

    def on_update(self, dt):
        # Handle the background animation
        if self.bg_anim is not None:
            self.bg_color.a = self.bg_anim.eval(self.time)
            if not self.bg_anim.is_active(self.time):
                self.bg_anim = None
                if self.anim_completion is not None:
                    self.anim_completion()

        if self.scale_anim is not None:
            new_scale = self.scale_anim.eval(self.time)
            self.scale.x = new_scale
            self.scale.y = new_scale
            if not self.scale_anim.is_active(self.time):
                self.scale_anim = None

        if self.alpha_anim is not None:
            self.shape.set_alpha(self.alpha_anim.eval(self.time))
            if not self.alpha_anim.is_active(self.time):
                self.alpha_anim = None

        self.time += dt

        # Handle new points
        if self.accepting_points:
            if self.end_mode == ShapeEditor.END_POSE:
                self.end_hold.on_update()

            new_pos = self.source()
            if self.end_mode == ShapeEditor.END_CLICK and new_pos is None:
                self.end_gesture(None)
                return

            # Only accept every third point (since data can be noisy)
            self.gesture_pos_idx = (self.gesture_pos_idx + 1) % 3
            if self.gesture_pos_idx == 0 and new_pos is not None:
                self.add_position(new_pos)

    def end_gesture(self, ignore):
        """
        Determines the final set of points to update the shape, and calls the
        on_complete handler.
        """
        if not self.accepting_points: return

        self.accepting_points = False

        # Update the shape
        point_array = np.array(self.shape.points)
        translation = np.array(self.translate.xy)
        scale = self.scale.x
        self.shape.set_points((point_array - self.shape_center) * scale + self.shape_center + translation)
        self.translate.xy = (0.0, 0.0)
        self.scale.x = 1.0
        self.scale.y = 1.0

        if self.end_mode == ShapeEditor.END_POSE:
            self.end_hold.set_enabled(False)
        self.on_complete(self, False)


    def hide_transition(self, callback, remove_object=False):
        """
        Performs a hide animation, and calls the given callback function on completion.
        If remove_object is True, animates the shape out of existence while
        fading out.
        """
        self.time = 0.0
        self.bg_anim = KFAnim((0.0, self.bg_color.a), (1.0, 0.0))
        if remove_object:
            self.scale_anim = KFAnim((0.0, self.scale.x), (0.5, self.scale.x * 2.0))
            self.alpha_anim = KFAnim((0.0, 1.0), (0.5, 0.0))
        self.anim_completion = callback
