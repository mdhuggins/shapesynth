from common.core import *
from common.gfxutil import *
from common.audio import *
from common.mixer import *
from common.synth import *
from common.clock import *

from kivy.core.image import Image
from kivy.graphics import Color, Line, Mesh, Translate, Scale, BindTexture, Rotate, Rectangle
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np

from color import ColorPalette
from composer import Conductor

class MeasureBar(InstructionGroup):

    def __init__(self, x_max, height, palette, scheduler):
        super(MeasureBar, self).__init__()
        self.x_max = x_max

        self.progress = 0
        self.height = height
        self.palette = palette
        self.scheduler = scheduler

        # self.hsv = (0, 1, 1)
        self.hsv = (0,0,1)
        self.front_color_anim = None
        self.back_color_anim = None

        self.back_color = Color(hsv=self.hsv)
        self.back_alpha = 0.2
        self.back_color.a =self.back_alpha
        self.add(self.back_color)

        self.back_line = Rectangle(pos=(0, 0), size=(self.x_max, self.height))
        self.add(self.back_line)

        self.third_color = Color(hsv=self.hsv)
        self.third_alpha = 0
        self.third_color.a = self.third_alpha
        self.add(self.third_color)

        self.third_line = Rectangle(pos=(0, 0), size=(self.x_max, self.height))
        self.add(self.third_line)

        self.third_line_anim = KFAnim((0,0.3), (Conductor.ticks_per_measure/4, 0))
        self.change_third_line_anim = KFAnim((0,0.8), (Conductor.ticks_per_measure/4, 0))
        self.changing = 0

        self.front_color = Color(hsv=self.hsv)
        self.front_alpha = 0.3
        self.front_color.a = self.front_alpha
        self.add(self.front_color)

        self.front_line = Rectangle(pos=(0, 0), size=(self.x_max * self.progress, self.height))
        self.add(self.front_line)

        self.alpha_increase = 0.5
        self.alpha_anim = None

        self.update_color(animated=False, init=True)


    def update_size(self, x_max, height=None):
        pass

    def set_progress(self, p):
        assert 0 <= p <= 1
        self.progress = p
        self.front_line.size=(self.x_max * self.progress, self.height)

    def update_color(self, animated=True, init=False):
        print("update color")
        """Called when a new harmony is chosen"""
        # if self.front_color_anim is not None: return

        tick = self.scheduler.get_tick()
        next_bar = quantize_tick_up(tick, Conductor.ticks_per_measure)
        ticks_to_next_bar = next_bar - tick

        new_hsv = self.palette.new_color(0.7)
        new_hsv = Color(hsv=new_hsv).rgb

        self.front_color_anim = (tick, KFAnim((0.0, *self.hsv), (ticks_to_next_bar if animated else 0.0, *new_hsv)))

        back_anim_ticks = min(Conductor.ticks_per_measure/16, ticks_to_next_bar)
        self.back_color_anim = (tick, KFAnim((0.0, *self.hsv), (back_anim_ticks if animated else 0.0, *new_hsv)))
        self.hsv = new_hsv

        alpha = self.back_color.a
        self.back_color.hsv = new_hsv
        self.back_color.a = alpha

        self.front_alpha_anim = (tick, KFAnim((0.0, 0),
                                        (back_anim_ticks if animated else 0.0, self.alpha_increase*0.75),
                                        (ticks_to_next_bar if animated else 0.0, self.alpha_increase)))

        self.back_alpha_anim = (tick, KFAnim((0.0, 0),
                                             (back_anim_ticks if animated else 0.0, self.alpha_increase * 0.75),
                                             (ticks_to_next_bar if animated else 0.0, self.alpha_increase),
                                             (ticks_to_next_bar+Conductor.ticks_per_measure/4, 0)))

        if not init:
            self.changing = 1

    def on_update(self):
        tick = self.scheduler.get_tick()
        next_bar = quantize_tick_up(tick, Conductor.ticks_per_measure)
        progress = 1-(next_bar-tick)/Conductor.ticks_per_measure
        self.set_progress(progress)
        # self.update_color()

        if self.front_color_anim is not None:
            start_tick, anim = self.front_color_anim
            new_color = anim.eval(tick - start_tick)

            alpha = self.front_color.a
            # self.front_color.hsv = new_color
            self.front_color.rgb = new_color
            self.third_color.rgb = new_color
            self.front_color.a = alpha

            if not anim.is_active(tick - start_tick):
                self.front_color_anim = None

        if self.back_color_anim is not None:
            start_tick, anim = self.back_color_anim
            new_color = anim.eval(tick - start_tick)

            alpha = self.back_color.a
            self.back_color.hsv = new_color
            self.back_color.rgb = new_color
            self.back_color.a = alpha

            if not anim.is_active(tick - start_tick):
                self.back_color_anim = None

        if self.front_alpha_anim:
            start_tick, anim = self.front_alpha_anim
            alpha_increase = anim.eval(tick - start_tick)

            # self.back_color.a = self.back_alpha + alpha_increase
            self.front_color.a = self.front_alpha + alpha_increase

            if not anim.is_active(tick - start_tick):
                self.front_alpha_anim = None
        else:
            # self.back_color.a = self.back_alpha
            self.front_color.a = self.front_alpha

        if self.back_alpha_anim:
            start_tick, anim = self.back_alpha_anim
            alpha_increase = anim.eval(tick - start_tick)

            self.back_color.a = self.back_alpha + alpha_increase
            # self.front_color.a = self.front_alpha + alpha_increase

            if not anim.is_active(tick - start_tick):
                self.back_alpha_anim = None
        else:
            self.back_color.a = self.back_alpha
            # self.front_color.a = self.front_alpha


        tick_progress = tick-(next_bar-Conductor.ticks_per_measure)

        if self.changing == 1 and self.third_line_anim.is_active(tick_progress):
            self.changing = 2

        if self.changing == 2:
            print("change anim")
            self.third_color.a = self.change_third_line_anim.eval(tick_progress)

            if not self.change_third_line_anim.is_active(tick_progress):
                self.changing = 0
        else:
            self.third_color.a = self.third_line_anim.eval(tick_progress)


