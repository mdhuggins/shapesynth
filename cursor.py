from kivy.clock import Clock as kivyClock
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Rectangle, Ellipse, Color, Fbo, ClearBuffers, ClearColor, Line, BindTexture, Translate, Scale
from kivy.graphics import PushMatrix, PopMatrix, Scale, Callback
from kivy.graphics.texture import Texture
from kivy.core.window import Window

import numpy as np

class AnimatedCursor(InstructionGroup):
    num_dots = 14
    num_clouds = 4
    dot_inertia = 0.6

    def __init__(self, source):
        super(AnimatedCursor, self).__init__()

        self.color = Color(hsv=(0.5, 0.7, 0.8))
        self.color.a = 0.2

        self.source = source
        self.is_visible = True

        self.dim = 120.0
        self.add(PushMatrix())
        self.translate = Translate(0, 0)
        self.scale = Scale(1)
        self.add(self.translate)
        self.add(self.scale)
        self.add(self.color)

        self.clouds = []
        cloud_positions = []
        cloud_velocities = []
        cloud_size = self.dim * 1.5
        for i in range(AnimatedCursor.num_clouds):
            spher_pos = (np.random.uniform(0.0, 2.0 * np.pi), np.random.uniform(0.0, np.pi))
            cloud_positions.append(spher_pos)
            velocity_range = 0.8
            cloud_velocities.append((np.random.uniform(-velocity_range, velocity_range), np.random.uniform(-velocity_range, velocity_range)))

            cloud = Rectangle(pos=(-cloud_size / 2.0, -cloud_size / 2.0), size=(cloud_size, cloud_size), source='res/blur_circle_more.png')
            self.add(cloud)
            self.clouds.append(cloud)

        self.cloud_positions = np.array(cloud_positions)
        self.cloud_velocities = np.array(cloud_velocities)

        self.back_translate = Translate(0, 0)
        self.add(self.back_translate)

        dot_positions = []
        dot_velocities = []
        self.dots = []
        self.dot_colors = []
        self.dot_size = 12.0
        for i in range(AnimatedCursor.num_dots):
            spher_pos = (np.random.uniform(0.0, 2.0 * np.pi), np.random.uniform(0.0, np.pi))
            dot_positions.append(spher_pos)
            velocity_range = 1.6
            dot_velocities.append((np.random.uniform(-velocity_range, velocity_range), np.random.uniform(-velocity_range, velocity_range)))

            pos = (np.random.uniform(-self.dim / 2, self.dim / 2), np.random.uniform(-self.dim / 2, self.dim / 2))
            dot = Ellipse(pos=(self.translate.x + pos[0] - self.dot_size / 2, self.translate.y + pos[1] - self.dot_size / 2), size=(self.dot_size, self.dot_size))
            dot_color = Color(hsv=(0.5, 0.8, 1.0))
            self.add(dot_color)
            self.add(dot)
            self.dots.append(dot)
            self.dot_colors.append(dot_color)

        self.dot_positions = np.array(dot_positions)
        self.dot_velocities = np.array(dot_velocities)
        self.add(PopMatrix())


    def set_pos(self, pos):
        if len(pos) > 2:
            new_scale = 0.5 + pos[2] * (1.5 - 0.5)
            self.scale.x = new_scale
            self.scale.y = new_scale
        self.translate.xy = pos[0:2]

    def set_visible(self, visible):
        changed = visible != self.is_visible
        if changed:
            self.is_visible = visible
            if not self.is_visible:
                self.color.a = 0.0
                for c in self.dot_colors:
                    c.a = 0.0
            else:
                self.color.a = 0.2

    def update_dot_positions(self):
        """Sets the dots' positions based on the self.dot_positions array."""
        r = self.dim / 2
        dot_x = r * np.sin(self.dot_positions[:,1]) * np.cos(self.dot_positions[:,0])
        dot_y = r * np.sin(self.dot_positions[:,1]) * np.sin(self.dot_positions[:,0])
        dot_z = r * np.cos(self.dot_positions[:,1])

        for dot, color, x, y, z in zip(self.dots, self.dot_colors, dot_x, dot_y, dot_z):
            size = (z + r) * self.dot_size / (2 * r)
            dot.pos = ((self.translate.x + x - size / 2) * (1 - AnimatedCursor.dot_inertia) + dot.pos[0] * AnimatedCursor.dot_inertia, (self.translate.y + y - size / 2) * (1 - AnimatedCursor.dot_inertia) + dot.pos[1] * AnimatedCursor.dot_inertia)
            dot.size = (size, size)
            color.a = (z + r) / (2 * r)

    def update_cloud_positions(self):
        """Sets the clouds' positions based on the self.cloud_positions array."""
        r = self.dim / 4.0
        cloud_x = r * np.sin(self.cloud_positions[:,1]) * np.cos(self.cloud_positions[:,0])
        cloud_y = r * np.sin(self.cloud_positions[:,1]) * np.sin(self.cloud_positions[:,0])

        for cloud, x, y in zip(self.clouds, cloud_x, cloud_y):
            cloud.pos = (x - cloud.size[0] / 2, y - cloud.size[1] / 2)

    def on_update(self, dt):
        """Updates the animated dots' positions."""
        pos = self.source()
        self.set_visible(pos is not None and (len(pos) <= 2 or pos[2] <= 0.45 or pos[1] >= 0.1))

        if not self.is_visible:
            return
        self.set_pos(pos)

        self.dot_positions = self.dot_positions + self.dot_velocities * dt
        self.cloud_positions = self.cloud_positions + self.cloud_velocities * dt
        #self.dot_positions[:,0] = np.mod(self.dot_positions[:,0], 2 * np.pi)
        #self.dot_positions[:,1] = np.mod(self.dot_positions[:,0], np.pi)
        self.back_translate.xy = (-self.translate.x, -self.translate.y)
        self.update_dot_positions()
        self.update_cloud_positions()
