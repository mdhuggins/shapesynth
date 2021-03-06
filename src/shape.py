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
from common.kivyparticle import ParticleSystem

import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import lfilter, butter
from triangulation import earclip
import matplotlib.pyplot as plt

from composer import *
from synth.shape_synth import *
from gesture import HoldGesture
from color import ColorPalette

import colorsys

class Shape(InstructionGroup):
    """
    Represents a 2D shape on the canvas.
    """

    hit_test_margin = 40.0

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
        self.has_initial_color = False
        self.shadow_index = len(self.children)
        self.color_frozen = False
        self.color_anim = None
        self.time = 0.0

        self.make_shape_properties()

        self.rotation = Rotate(0.0)
        self.rotation_rate = np.random.uniform(-15.0, 15.0)
        self.add(PushMatrix())
        self.translate = Translate(*self.screen_center)
        self.add(self.translate)

        self.ps = ParticleSystem('res/particle_explosion/particle.pex')
        self.add(self.ps)

        self.add(self.rotation)
        self.scale = Scale(1.0, 1.0)
        self.add(self.scale)
        self.back_translate = Translate(*(-self.screen_center))
        self.add(self.back_translate)
        self.fill_color = Color(*colorsys.hsv_to_rgb(*self.hsv))
        self.fill_color.a = 0.5
        self.add(self.fill_color)
        self.mesh = Mesh(mode='triangles')
        self.update_mesh()
        self.add(self.mesh)
        self.stroke_color = Color(*self.fill_color.rgb)
        self.add(self.stroke_color)
        self.curve = Line(points=[coord for point in self.points for coord in point], segments=20 * len(self.points), loop=True)
        self.curve.width = 3.0
        self.add(self.curve)
        self.add(PopMatrix())

        self.shadow_reenable_time = 0
        self.colors = [self.fill_color, self.stroke_color]
        self.shadow_anims = {}

        self.make_synth()
        self.make_composer(sched, mixer)
        self.update_particle_system()

    def set_points(self, points):
        """Takes the given numpy nx2 array of points and sets the shape's points
        accordingly."""
        self.points = [tuple(row) for row in points]
        self.curve.points = [coord for point in self.points for coord in point]
        self.update_mesh()
        self.make_shape_properties()
        self.translate.xy = self.screen_center
        self.back_translate.xy = -self.screen_center

        for (shadow, color) in self.shadow_anims:
            shadow.pos = self.screen_center - np.array(shadow.size) / 2.0

    def update_for_window_size(self, last_size):
        """Updates this shape's points using the given size as the last size."""
        self.set_points(np.array(self.points) * np.array([Window.width, Window.height]) / np.array(last_size))

    def set_alpha(self, alpha):
        self.fill_color.a = alpha / 2.0
        self.stroke_color.a = alpha

    def hit_test(self, point):
        """
        Returns True if the given point is contained within the mesh.
        """
        dist = np.linalg.norm(point[:2] - self.screen_center)
        return dist <= self.diameter / 1.414 + Shape.hit_test_margin

    def update_mesh(self):
        """
        Sets the Mesh's vertices and indices to display this shape.
        """
        vertices = []
        theta = 2 * np.pi / len(self.points)
        for point in self.points:
            vertices += [point[0], point[1], 0, 0]

        # Use earclipping algorithm to triangulate the shape
        triangles = earclip(self.points)
        indices = []
        for point1, point2, point3 in triangles:
            indices += [self.points.index(point1), self.points.index(point2), self.points.index(point3)]

        self.mesh.vertices = vertices
        self.mesh.indices = indices

    def update_sound(self, new_composer=True):
        """
        Refreshes the sonic properties of the shape.
        """
        was_on = self.composer.playing
        if new_composer:
            self.composer.stop()
        self.make_shape_properties()
        self.make_synth()
        self.make_composer(self.composer.sched, self.composer.mixer, was_on, new_composer)
        self.update_particle_system()

    def make_shape_properties(self):
        """
        Computes properties about this shape that can be used to make a synth
        and/or composer.
        """
        point_array = np.array(self.points)
        self.screen_center = np.mean(point_array, axis=0)
        self.center = self.screen_center / np.array([Window.width, Window.height])
        self.min_x = np.min(point_array[:,0])
        self.max_x = np.max(point_array[:,0])
        self.min_y = np.min(point_array[:,1])
        self.max_y = np.max(point_array[:,1])
        self.area = (self.max_x - self.min_x) * (self.max_y - self.min_y)
        self.diameter = np.linalg.norm(np.array([self.max_x, self.max_y]) - np.array([self.min_x, self.min_y]))

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

    def make_synth(self):
        """
        Creates a ShapeSynth for this shape. TODO: Use properties other than
        center and area.
        """
        min_gain = 0.05
        max_gain = 0.6
        window_area = Window.width * Window.height
        gain = np.clip(self.area / (0.0125 * window_area) * (max_gain - min_gain) + min_gain, min_gain, max_gain)

        self.synth = ShapeSynth(self.center[0], self.center[1], gain, self.roughness)

    def make_composer(self, sched, mixer, start=True, reset=True):
        """
        Builds this shape's Composer using its location and properties of its
        vertices.
        """
        if reset:
            self.composer = Composer(sched, mixer, self.synth.make_note)
        self.composer.pitch_level = (1/16 + self.center[1]*(1-2*self.center[0])/8 + np.sqrt(self.center[0] * 0.7))*8/9
        self.composer.pitch_variance = (self.center[0] / 2.0) ** 2
        self.composer.complexity = self.roughness
        self.composer.harmonic_obedience = np.sqrt(1.0 - self.roughness)
        self.composer.bass_preference = 1 - self.center[0]
        self.composer.arpeggio_preference = 2.0 * self.center[0] * (1 - self.center[0])
        self.composer.update_interval = 4 if self.center[0] > 0.3 else 8
        self.composer.velocity_level = 0.5
        self.composer.velocity_variance = self.center[0] * (1 - self.center[0])
        self.composer.update_callback = self.update_color
        self.composer.on_note = self.on_note

        if start:
            self.composer.start()

    def update_particle_system(self):
        """
        Updates the particle system's properties based on the current shape
        location and size.
        """
        self.ps.emitter_x = 0.0
        self.ps.emitter_y = 0.0
        self.ps.speed = 100.0 * np.sqrt(self.area / 7000.0)
        self.ps.radial_acceleration = -self.ps.speed * 0.7
        r, g, b, a = self.ps.start_color

    def on_note(self, pitch, velocity, dur):
        """Called when the ShapeSynth plays a note."""
        if self.color_frozen: return

        center = ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)
        dim = max(self.max_x - self.min_x, self.max_y - self.min_y) * 2.5

        self.ps.start()
        def stop(_):
            self.ps.stop()
        kivyClock.schedule_once(stop, 0.4)

        if self.time < self.shadow_reenable_time: return

        new_circle = Rectangle(pos=(center[0] - dim / 2, center[1] - dim / 2), size=(dim, dim))
        #new_circle = CEllipse(cpos=center, csize=(dim, dim), texture=tex)
        color = Color(*self.fill_color.rgb)
        color.a = 0.3
        self.colors.append(color)
        circle_duration = min(max(2.0 * dur, 2.0), 8.0)
        final_size = dim * (2.0 + 8.0 * (self.center[1] + 0.4) / 1.8)
        self.shadow_anims[(new_circle, color)] = (self.time, KFAnim((0.0, dim), (circle_duration, final_size)), KFAnim((0.0, color.a), (circle_duration, 0.0)))
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
                color.rgb = new_color
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

    def update_color(self):
        """Called when the composer generates new notes."""
        if self.color_frozen: return
        if self.color_anim is not None: return

        new_hsv = self.palette.new_color(self.composer.pitch_level)
        r, g, b = colorsys.hsv_to_rgb(*new_hsv)
        if not self.has_initial_color:
            # Don't animate the first time
            for color in self.colors:
                old_a = color.a
                color.rgb = (r, g, b)
                color.a = old_a
            self.has_initial_color = True
        else:
            self.color_anim = (self.time, KFAnim((0.0, *self.fill_color.rgb), (1.0, r, g, b)))
        self.hsv = new_hsv

        if self.ps is not None:
            self.ps.start_color = (r, g, b, 0.4)
            self.ps.end_color = (r, g, b, 0.0)

SHAPE_CLOSE_THRESHOLD = 40
MAX_DISTANCE_THRESHOLD = 80

POINT_QUERY_INTERVAL = 2

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
        self.bg_color = Color(0, 0, 0)
        self.bg_color.a = 0.0
        self.bg_anim = KFAnim((0.0, 0.0), (0.5, 0.7))
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
            elif dist < SHAPE_CLOSE_THRESHOLD / 2 and self.max_distance <= MAX_DISTANCE_THRESHOLD and dist <= self.max_distance and len(self.points) > 25:
                self.accepting_points = False
                self.on_complete([])
            elif dist > self.max_distance:
                self.max_distance = dist

    def smooth_shape(self):
        """Updates the points to be smoother and form a closed boundary."""
        # Apply LPF to the points to smooth out the shape
        points = np.array([(self.points[i], self.points[i + 1]) for i in range(0, len(self.points), 2)])
        filter = butter(4, 0.4, btype='lowpass')
        # LPF the full cycle, so we're sure the shape is contiguous
        point_x = lfilter(*filter, np.concatenate([points[:,0], points[:,0]]))[len(points) // 2:len(points) + len(points) // 2 + 1]
        point_y = lfilter(*filter, np.concatenate([points[:,1], points[:,1]]))[len(points) // 2:len(points) + len(points) // 2 + 1]
        point_indexes = list(range(len(point_x) // 2, len(point_x))) + list(range(len(point_x) // 2 + 1))
        new_points = [coord for i in point_indexes for coord in [point_x[i], point_y[i]]]

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
                    return False

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
            if new_pos is None or ((len(new_pos) > 2 and (new_pos[2] > 0.7 or new_pos[1] <= 0.1)) or len(self.points) > 800):
                # The user stopped drawing - cancel
                self.accepting_points = False
                self.on_complete([])
                return
            # Only accept every third point (since data can be noisy)
            self.gesture_pos_idx = (self.gesture_pos_idx + 1) % POINT_QUERY_INTERVAL
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
    offscreen_threshold = 5.0

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
        self.bg_color = Color(0, 0, 0)
        self.bg_color.a = 0.0
        self.bg_anim = KFAnim((0.0, 0.0), (0.5, 0.7))
        self.add(self.bg_color)
        self.add(Rectangle(pos=(0,0), size=(Window.width, Window.height)))
        self.shape_center = self.shape.screen_center

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

        self.move_refresh_rate = 0.25  # In seconds
        self.move_clock = 0


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
        if self.will_remove(pos):
            # Remove the shape
            self.accepting_points = False
            self.on_complete(self, True)
            return
        if len(pos) > 2:
            new_scale = 1 + (pos[2] - self.original_position[2])
            self.scale.x = new_scale
            self.scale.y = new_scale

    def get_current_pos(self):
        return self.current_position

    def will_remove(self, pos):
        """Returns True if the given position represents a shape deletion gesture."""
        if self.end_mode == ShapeEditor.END_POSE:
            return pos[2] > 0.8 and self.last_position is not None and (pos[2] - self.last_position[2]) > 0.2
        else:
            return pos[0] < ShapeEditor.offscreen_threshold or pos[0] > Window.width - ShapeEditor.offscreen_threshold or pos[1] < ShapeEditor.offscreen_threshold or pos[1] > Window.height - ShapeEditor.offscreen_threshold

    def on_update(self, dt):
        # Handle the background animation
        if self.bg_anim is not None:
            self.bg_color.a = self.bg_anim.eval(self.time)
            if not self.bg_anim.is_active(self.time):
                self.bg_anim = None
                if self.anim_completion is not None:
                    self.anim_completion()
                    return False

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
            self.gesture_pos_idx = (self.gesture_pos_idx + 1) % POINT_QUERY_INTERVAL
            if self.gesture_pos_idx == 0 and new_pos is not None:
                self.add_position(new_pos)

            # Update composer/synth
            self.move_clock += dt
            if self.move_clock > self.move_refresh_rate:
                self.move_clock = 0
                self.on_move()

    def on_move(self):
        # Update the shape
        self.original_position = self.current_position

        point_array = np.array(self.shape.points)
        translation = np.array(self.translate.xy)
        scale = self.scale.x
        self.shape.set_points((point_array - self.shape_center) * scale + self.shape_center + translation)
        self.translate.xy = (0.0, 0.0)
        self.scale.x = 1.0
        self.scale.y = 1.0
        self.shape_center = self.shape.screen_center
        self.scale.origin = self.shape_center


        self.shape.make_shape_properties()
        self.shape.make_synth()
        self.shape.update_sound(new_composer=False)
        self.shape.composer.note_factory = self.shape.synth.make_note
        self.shape.composer.clear_notes()


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
