from kivy.clock import Clock as kivyClock
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Rectangle, Ellipse, Color, Fbo, ClearBuffers, ClearColor, Line, BindTexture, Translate, Scale
from kivy.graphics import PushMatrix, PopMatrix, Scale, Callback
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from common.gfxutil import KFAnim

import numpy as np

class Grid(InstructionGroup):
    """
    Handles drawing a grid over the window. By default the grid is invisible; it
    can be made visible by calling set_grid_visible.
    """
    grid_alpha = 0.6
    alpha_anim_dur = 0.4

    def __init__(self, grid_interval=160):
        """
        grid_interval: the number of pixels between grid lines
        """
        super(Grid, self).__init__()
        self.grid_interval = grid_interval
        self.alpha_anim = None
        self.grid_color = Color(1, 1, 1)
        self.grid_color.a = 0.0
        self.add(self.grid_color)
        self.make_grid()
        self.grid_visible = False

        self.time = 0.0

    def make_grid(self):
        """
        Creates the lines that make up the grid.
        """
        self.add(PushMatrix())
        self.add(Translate(Window.width / 2.0, Window.height / 2.0))

        x_bound = int(np.ceil(Window.width / 2.0 / self.grid_interval))
        y_bound = int(np.ceil(Window.height / 2.0 / self.grid_interval))
        for x in range(-x_bound, x_bound + 1):
            line = Line(points=[x * self.grid_interval, -Window.height / 2.0, x * self.grid_interval, Window.height / 2.0])
            line.width = 0.5
            self.add(line)

        for y in range(-y_bound, y_bound + 1):
            line = Line(points=[-Window.width / 2.0, y * self.grid_interval, Window.width / 2.0, y * self.grid_interval])
            line.width = 0.5
            self.add(line)

        self.add(PopMatrix())

    def set_grid_visible(self, visible):
        """
        Sets whether the grid is visible with an animation.
        """
        if visible and not self.grid_visible:
            self.alpha_anim = (self.time, KFAnim((0.0, self.grid_color.a), (Grid.alpha_anim_dur, Grid.grid_alpha)))
            self.grid_visible = True
        elif not visible and self.grid_visible:
            self.alpha_anim = (self.time, KFAnim((0.0, self.grid_color.a), (Grid.alpha_anim_dur, 0.0)))
            self.grid_visible = False

    def on_update(self, dt):
        if self.alpha_anim is not None:
            start_time, anim = self.alpha_anim
            self.grid_color.a = anim.eval(self.time - start_time)
            if not anim.is_active(self.time - start_time):
                self.alpha_anim = None

        self.time += dt
