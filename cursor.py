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

    normal_alpha = 0.2
    normal_cloud_scale = 1.0
    drawing_cloud_scale = 0.3

    normal_radius_proportion = 0.5
    drawing_radius_proportion = 0.05

    normal_shadow_alpha = 0.4
    drawing_shadow_alpha = 0.0

    NORMAL = 'normal'
    HOLDING = 'holding'
    DRAWING = 'drawing'
    EDITING = 'editing'

    def __init__(self, source, dim, normal_hsv=(0.5, 0.7, 0.8), drawing_hsv=(0.5, 0.7, 0.8), editing_hsv=(0.5, 0.7, 0.8)):
        """
        source - a function that, when called, returns a new cursor point or None.
        dim - the average size of the cursor
        (normal/drawing/editing)_hsv - an HSV tuple for the color of the cursor
            at each state
        """
        super(AnimatedCursor, self).__init__()

        self.normal_hsv = normal_hsv
        self.drawing_hsv = drawing_hsv
        self.editing_hsv = editing_hsv

        self.color = Color(hsv=self.normal_hsv)
        self.color.a = AnimatedCursor.normal_alpha

        self.source = source
        self.is_visible = True

        self.state = AnimatedCursor.NORMAL
        self.last_state = AnimatedCursor.NORMAL
        self.theta_velocity = 0.0
        self.phi_velocity = 0.0
        self.radius_proportion = AnimatedCursor.normal_radius_proportion

        self.dim = dim
        self.add(PushMatrix())
        self.translate = Translate(0, 0)
        self.add(self.translate)

        self.add(PushMatrix())
        self.scale = Scale(1)
        self.add(self.scale)

        # Shadow
        self.shadow_color = Color(0.0, 0.0, 0.0)
        self.shadow_color.a = AnimatedCursor.normal_shadow_alpha
        shadow_size = self.dim * 4.0
        self.add(self.shadow_color)
        self.shadow = Rectangle(pos=(-shadow_size / 2.0, -shadow_size / 2.0), size=(shadow_size, shadow_size), source='res/blur_circle.png')
        self.add(self.shadow)

        self.add(self.color)
        self.add(PushMatrix())

        # Clouds
        self.cloud_scale = Scale(1)
        self.add(self.cloud_scale)

        self.clouds = []
        cloud_positions = []
        cloud_velocities = []
        cloud_size = self.dim * 1.5
        for i in range(AnimatedCursor.num_clouds):
            spher_pos = (np.random.uniform(0.0, 2.0 * np.pi), np.random.uniform(0.0, np.pi))
            cloud_positions.append(spher_pos)
            velocity_range = 0.8
            cloud_velocities.append((np.random.uniform(-velocity_range, velocity_range), np.random.uniform(-velocity_range, velocity_range)))

            cloud = Rectangle(pos=(-cloud_size / 2.0, -cloud_size / 2.0), size=(cloud_size, cloud_size), source='res/blur_circle.png')
            self.add(cloud)
            self.clouds.append(cloud)

        self.cloud_positions = np.array(cloud_positions)
        self.cloud_velocities = np.array(cloud_velocities)

        self.add(PopMatrix()) # pop cloud scale
        self.add(PopMatrix()) # pop overall scale
        self.back_translate = Translate(0, 0)
        self.add(self.back_translate)

        # Dots
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
            dot_color = Color(hsv=self.normal_hsv)
            self.add(dot_color)
            self.add(dot)
            self.dots.append(dot)
            self.dot_colors.append(dot_color)

        self.dot_positions = np.array(dot_positions)
        self.dot_velocities = np.array(dot_velocities)
        self.add(PopMatrix())

    def set_state(self, state):
        self.state = state

    def set_pos(self, pos):
        if len(pos) > 2:
            new_scale = 0.5 + np.sqrt(pos[2]) * (1.5 - 0.5)
            self.scale.x = new_scale
            self.scale.y = new_scale
            self.set_hsv((self.normal_hsv[0], 1 - pos[2], self.normal_hsv[2]))
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
                self.color.a = AnimatedCursor.normal_alpha

    def update_dot_positions(self):
        """Sets the dots' positions based on the self.dot_positions array."""
        r = self.dim * self.radius_proportion
        if self.is_in_control_state():
            inertia = AnimatedCursor.dot_inertia * 1.3
        else:
            inertia = AnimatedCursor.dot_inertia

        dot_x = r * np.sin(self.dot_positions[:,1]) * np.cos(self.dot_positions[:,0])
        dot_y = r * np.sin(self.dot_positions[:,1]) * np.sin(self.dot_positions[:,0])
        dot_z = r * np.cos(self.dot_positions[:,1])

        for dot, color, x, y, z in zip(self.dots, self.dot_colors, dot_x, dot_y, dot_z):
            size = (z + r) * self.dot_size / (2 * r)
            dot.pos = ((self.translate.x + x - size / 2) * (1 - inertia) + dot.pos[0] * inertia, (self.translate.y + y - size / 2) * (1 - inertia) + dot.pos[1] * inertia)
            dot.size = (size, size)
            color.a = (z + r) / (2 * r)

    def update_cloud_positions(self):
        """Sets the clouds' positions based on the self.cloud_positions array."""
        r = self.dim / 5.0
        cloud_x = r * np.sin(self.cloud_positions[:,1]) * np.cos(self.cloud_positions[:,0])
        cloud_y = r * np.sin(self.cloud_positions[:,1]) * np.sin(self.cloud_positions[:,0])

        for cloud, x, y in zip(self.clouds, cloud_x, cloud_y):
            cloud.pos = (x - cloud.size[0] / 2, y - cloud.size[1] / 2)

    def on_update(self, dt):
        """Updates the animated dots' positions."""
        pos = self.source()
        self.set_visible(pos is not None and (len(pos) <= 2 or pos[2] <= 0.4 or pos[1] >= 0.1))

        if not self.is_visible:
            return
        self.set_pos(pos)

        if self.last_state != self.state:
            self.handle_change_state()
        else:
            if self.state == AnimatedCursor.HOLDING:
                self.dot_positions[:,0] -= self.theta_velocity * dt
                self.theta_velocity += 3.0 * dt
                self.dot_positions[:,1] = np.maximum(self.dot_positions[:,1] + self.phi_velocity * dt, 0.0)
                self.color.a = min(self.color.a + 0.4 * dt, 0.6)
                self.phi_velocity -= 2.0 * dt

                # update cloud scale
                new_scale = max(self.cloud_scale.x * 0.95, AnimatedCursor.drawing_cloud_scale)
                self.cloud_scale.x = new_scale
                self.cloud_scale.y = new_scale
            elif self.is_in_control_state():
                self.dot_positions[:,0] = self.dot_positions[:,0] + self.dot_velocities[:,0] * dt * 2.0
                self.dot_positions[:,1] = np.sin(np.pi / 3.0 * (np.arange(len(self.dot_positions)) - 2.0 * np.pi / len(self.dot_positions))) * np.pi / 16.0
                self.cloud_positions = self.cloud_positions + self.cloud_velocities * dt
            else:
                self.dot_positions = self.dot_positions + self.dot_velocities * dt
                self.cloud_positions = self.cloud_positions + self.cloud_velocities * dt

            if not self.is_in_control_state() and self.cloud_scale.x < AnimatedCursor.normal_cloud_scale:
                new_scale = min(self.cloud_scale.x * 1.02, AnimatedCursor.normal_cloud_scale)
                self.cloud_scale.x = new_scale
                self.cloud_scale.y = new_scale

        self.back_translate.xy = (-self.translate.x, -self.translate.y)
        self.update_dot_positions()
        self.update_cloud_positions()

    def is_in_control_state(self, state=None):
        """Returns if the state is currently DRAWING or EDITING. If `state` is
        not None, computes the control status of that state."""
        return (state if state is not None else self.state) in [AnimatedCursor.DRAWING, AnimatedCursor.EDITING]

    def set_hsv(self, hsv):
        """Sets the given HSV on all colors associated with this cursor."""
        for color in [self.color] + self.dot_colors:
            old_alpha = color.a
            color.hsv = hsv
            color.a = old_alpha

    def handle_change_state(self):
        """Updates the dot positions given that the last state is different."""
        if self.state == AnimatedCursor.HOLDING:
            # Round all positions to their nearest position along the circle
            self.dot_positions[:,0] = np.linspace(0.0, np.pi * 2, AnimatedCursor.num_dots, endpoint=False)
            self.dot_positions[:,1] = np.pi / 3.0
            self.theta_velocity = 0.1
            self.phi_velocity = -0.5
        elif self.state == AnimatedCursor.NORMAL and self.last_state == AnimatedCursor.HOLDING:
            self.theta_velocity = 0.0
            self.phi_velocity = 0.0
            self.color.a = AnimatedCursor.normal_alpha
        elif not self.is_in_control_state() and self.is_in_control_state(self.last_state):
            self.shadow_color.a = AnimatedCursor.normal_shadow_alpha
            self.set_hsv(self.normal_hsv)
        elif self.state == AnimatedCursor.DRAWING and not self.is_in_control_state(self.last_state):
            self.shadow_color.a = AnimatedCursor.drawing_shadow_alpha
            self.set_hsv(self.drawing_hsv)
        elif self.state == AnimatedCursor.EDITING and not self.is_in_control_state(self.last_state):
            self.shadow_color.a = AnimatedCursor.drawing_shadow_alpha
            self.set_hsv(self.editing_hsv)
        self.last_state = self.state
