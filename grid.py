from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Line, Translate, PushMatrix, PopMatrix
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

    def __init__(self, grid_interval=240):
        """
        grid_interval: the number of pixels between grid lines
        """
        super(Grid, self).__init__()
        self.grid_interval = grid_interval
        self.alpha_anim = None
        self.target_anims = None
        self.target_lines = None
        self.grid_color = Color(hsv=(0.5, 0.7, 0.4))
        self.grid_color.a = 0.0
        self.add(self.grid_color)
        self.make_grid()
        self.grid_visible = False

        self.time = 0.0

    def make_grid(self):
        """
        Creates the lines that make up the grid.
        """
        self.gridlines = []
        self.add(PushMatrix())
        self.add(Translate(Window.width / 2.0, Window.height / 2.0))

        x_bound = int(np.ceil(Window.width / 2.0 / self.grid_interval))
        y_bound = int(np.ceil(Window.height / 2.0 / self.grid_interval))
        for x in range(-x_bound, x_bound + 1):
            line = Line(points=[x * self.grid_interval, -Window.height / 2.0, x * self.grid_interval, Window.height / 2.0])
            line.width = 0.5
            self.add(line)
            self.gridlines.append(line)

        for y in range(-y_bound, y_bound + 1):
            line = Line(points=[-Window.width / 2.0, y * self.grid_interval, Window.width / 2.0, y * self.grid_interval])
            line.width = 0.5
            self.add(line)
            self.gridlines.append(line)

        self.add(PopMatrix())

    def redraw_grid(self):
        """
        Removes the existing grid and draws a new one.
        """
        for line in self.gridlines:
            self.remove(line)
        self.make_grid()

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

    def make_target_animation(self, target, duration):
        """
        Creates and starts an animation that points toward the given target point.
        """
        if self.target_lines is not None or self.target_anims is not None:
            return

        self.target_lines = [
            Line(points=[target[0], 0, target[0], 0]),
            Line(points=[target[0], Window.height, target[0], Window.height]),
            Line(points=[0, target[1], 0, target[1]]),
            Line(points=[Window.width, target[1], Window.width, target[1]])
        ]
        end_time = duration
        complete_time = duration * 0.6
        self.target_anims = (self.time, [
            (KFAnim((0, target[0])),
             KFAnim((0, 0), (complete_time, 0), (end_time, target[1])),
             KFAnim((0, target[0])),
             KFAnim((0, 0), (complete_time, target[1])),
             KFAnim((0, 0.5), (complete_time, 3.0))),
            (KFAnim((0, target[0])),
             KFAnim((0, Window.height), (complete_time, Window.height), (end_time, target[1])),
             KFAnim((0, target[0])),
             KFAnim((0, Window.height), (complete_time, target[1])),
             KFAnim((0, 0.5), (complete_time, 3.0))),
            (KFAnim((0, 0), (complete_time, 0), (end_time, target[0])),
             KFAnim((0, target[1])),
             KFAnim((0, 0), (complete_time, target[0])),
             KFAnim((0, target[1])),
             KFAnim((0, 0.5), (complete_time, 3.0))),
            (KFAnim((0, Window.width), (complete_time, Window.width), (end_time, target[0])),
             KFAnim((0, target[1])),
             KFAnim((0, Window.width), (complete_time, target[0])),
             KFAnim((0, target[1])),
             KFAnim((0, 0.5), (complete_time, 3.0)))
        ])
        for line in self.target_lines:
            line.width = 0.5
            self.add(line)

    def on_update(self, dt):
        if self.alpha_anim is not None:
            start_time, anim = self.alpha_anim
            self.grid_color.a = anim.eval(self.time - start_time)
            if not anim.is_active(self.time - start_time):
                self.alpha_anim = None

        if self.target_anims is not None:
            start_time = self.target_anims[0]
            is_active = False
            for line, anims in zip(self.target_lines, self.target_anims[1]):
                line.points = [anim.eval(self.time - start_time) for anim in anims[:4]]
                #print(line.points)
                line.width = anims[-1].eval(self.time - start_time)
                is_active = is_active or any(anim.is_active(self.time - start_time) for anim in anims)
            if not is_active:
                self.target_anims = None
                for line in self.target_lines:
                    self.remove(line)
                self.target_lines = None

        self.time += dt
