from common.core import *
from common.gfxutil import *
from common.audio import *
from common.mixer import *
from common.synth import *
from common.clock import *

from kivy.graphics import Color, Line, Mesh
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np
from scipy.spatial import Delaunay
from scipy.signal import lfilter, butter
import tripy
import matplotlib.pyplot as plt

from composer import *
from synth.shape_synth import *

class Shape(InstructionGroup):
    """
    Represents a 2D shape on the canvas.
    """

    def __init__(self, points, color, sched, mixer):
        """
        points - a list of points bounding the shape
        color - a tuple (H, S, V) representing the shape's color
        sched - a scheduler on which to build the composer
        mixer - the mixer to which to add notes
        """
        super(Shape, self).__init__()
        self.points = points
        self.hsv = color
        self.fill_color = Color(hsv=self.hsv)
        self.fill_color.a = 0.5
        self.add(self.fill_color)
        self.mesh = self.make_mesh()
        self.add(self.mesh)
        self.stroke_color = Color(hsv=self.hsv)
        self.add(self.stroke_color)
        self.curve = Line(points=[coord for point in self.points for coord in point], segments=20 * len(self.points), loop=True)
        self.curve.width = 3.0
        self.add(self.curve)

        self.make_composer(sched, mixer)

    def make_mesh(self):
        """
        Builds the Mesh needed to display this shape.
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
        return Mesh(vertices=vertices, indices=indices, mode='triangles')

    def make_composer(self, sched, mixer):
        """
        Builds this shape's Composer using its location and properties of its
        vertices.
        """
        point_array = np.array(self.points)
        center = np.mean(point_array, axis=0) / np.array([Window.width, Window.height])
        area = (np.max(point_array[:,0]) - np.min(point_array[:,0])) * (np.max(point_array[:,1]) - np.min(point_array[:,1]))
        min_gain = 0.05
        max_gain = 0.3
        gain = np.clip(area / 10000.0 * (max_gain - min_gain) + min_gain, min_gain, max_gain)

        self.synth = ShapeSynth(center[0], center[1], gain)
        self.composer = Composer(sched, mixer, self.synth.make_note,
                                 np.sqrt(center[0] * 0.7), # pitch range
                                 (center[0] / 2.0) ** 2, # pitch variance
                                 center[0] ** 6, # complexity
                                 np.sqrt(1.0 - center[0]), # harmonic obedience
                                 4) # number of beats to generate
        self.composer.start()

    # def make_simple_note(self, pitch, dur):
    #     note_gen = NoteGenerator(int(pitch), 0.1)
    #     env_params = Envelope.magic_envelope(0.6)
    #     return Envelope(note_gen, *env_params)


SHAPE_CLOSE_THRESHOLD = 20
MAX_DISTANCE_THRESHOLD = 30

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
        self.time += dt

        # Handle new points
        if self.accepting_points:
            new_pos = self.source()
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
        self.shape_anim_points = (np.array(self.points), np.array(final_points))
        self.shape_timing = spring_timing_function(1.0)
        self.anim_completion = callback
