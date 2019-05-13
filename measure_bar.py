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

    def __init__(self, x_max, height, palette, scheduler, left_label, right_label):
        super(MeasureBar, self).__init__()
        self.x_max = x_max

        self.progress = 0
        self.height = height
        self.palette = palette
        self.scheduler = scheduler

        self.hsv = (0,0,1)

        # Back bar
        self.back_color = Color(hsv=self.hsv)
        self.back_alpha = 0.2
        self.back_color.a =self.back_alpha
        self.add(self.back_color)

        self.back_line = Rectangle(pos=(0, 0), size=(self.x_max, self.height))
        self.add(self.back_line)

        self.back_color_anim = None


        # Third bar
        self.third_color = Color(hsv=self.hsv)
        self.third_alpha = 0
        self.third_color.a = self.third_alpha
        self.add(self.third_color)

        self.third_line = Rectangle(pos=(0, 0), size=(self.x_max, self.height))
        self.add(self.third_line)

        self.third_line_anim = KFAnim((0,0.3), (Conductor.ticks_per_measure/4, 0))
        self.change_third_line_anim = KFAnim((0,0.8), (Conductor.ticks_per_measure/4, 0))
        self.changing = 0  # 0 not changing, 1 changing, 2 just changed

        # Front bar
        self.front_color = Color(hsv=self.hsv)
        self.front_alpha = 0.3
        self.front_color.a = self.front_alpha
        self.add(self.front_color)

        self.front_line = Rectangle(pos=(0, 0), size=(self.x_max * self.progress, self.height))
        self.add(self.front_line)

        self.front_color_anim = None

        # Front/back alpha anim
        self.alpha_increase = 0.5
        self.alpha_anim = None

        # Labels
        self.left_label = left_label
        self.left_label.pos = (40.0, self.height + 20.0)
        self.left_label.font_size = '16sp'
        self.left_label.halign = 'left'
        self.left_label.valign = 'bottom'
        self.left_label.text = 'Harmony'
        self.left_label.bind(texture_size=self.left_label.setter('size'))
        self.left_label.size_hint = (None, None)
        self.right_label = right_label
        self.right_label.font_size = '16sp'
        self.right_label.halign = 'right'
        self.right_label.valign = 'bottom'
        self.right_label.text = 'C, D, F#, A'
        self.right_label.size_hint = (None, None)
        self.right_label.bind(texture_size=self.right_label.setter('size'))
        self.right_label.pos = (Window.width - 40.0 - 180.0, self.height + 20.0)

        self.left_label_alpha = None
        self.right_label_alpha = None
        self.left_label_slide = None
        self.right_label_slide = None

        self.update_color(animated=False, init=True)


    def update_size(self, x_max, height=None):
        self.x_max = x_max
        if height:
            self.height = height

        self.front_line.size = (self.x_max * self.progress, self.height)
        self.third_line.size = (self.x_max, self.height)
        self.back_line.size = (self.x_max, self.height)
        self.right_label.pos = (x_max - 40.0 - 180.0, self.height + 20.0)


    def set_progress(self, p):
        assert 0 <= p <= 1
        self.progress = p
        self.front_line.size=(self.x_max * self.progress, self.height)

    def update_color(self, animated=True, init=False):
        """Called when a new harmony is chosen"""
        # if self.front_color_anim is not None: return

        tick = self.scheduler.get_tick()
        next_bar = quantize_tick_up(tick, Conductor.ticks_per_measure)
        ticks_to_next_bar = next_bar - tick

        new_hsv = self.palette.new_color(0.7)
        new_rgb = Color(hsv=new_hsv).rgb

        self.front_color_anim = (tick, KFAnim((0.0, *self.hsv), (ticks_to_next_bar if animated else 0.0, *new_rgb)))

        back_anim_ticks = min(Conductor.ticks_per_measure/16, ticks_to_next_bar)
        self.back_color_anim = (tick, KFAnim((0.0, *self.hsv), (back_anim_ticks if animated else 0.0, *new_rgb)))
        self.hsv = new_rgb

        alpha = self.back_color.a
        self.back_color.hsv = new_rgb
        self.back_color.a = alpha

        self.front_alpha_anim = (tick, KFAnim((0.0, 0),
                                        (back_anim_ticks if animated else 0.0, self.alpha_increase*0.75),
                                        (ticks_to_next_bar if animated else 0.0, self.alpha_increase)))

        self.back_alpha_anim = (tick, KFAnim((0.0, 0),
                                             (back_anim_ticks if animated else 0.0, self.alpha_increase * 0.75),
                                             (ticks_to_next_bar if animated else 0.0, self.alpha_increase),
                                             (ticks_to_next_bar+Conductor.ticks_per_measure/4, 0)))

        # Update labels
        if init:
            self.left_label.text = Conductor.harmony_string()
            self.left_label.color = (*new_rgb, 1)
            self.right_label.text = ''
        elif self.right_label_slide is not None:
            if self.left_label.text == self.right_label.text:
                # The left label was already updated - update it
                self.left_label.text = Conductor.harmony_string()
                self.left_label.color = (*new_rgb, 1)

            self.right_label.text = Conductor.harmony_string()
            self.right_label.color = (*new_rgb, 1)
        else:
            # update right label immediately
            self.right_label.text = Conductor.harmony_string()
            self.right_label.color = (*new_rgb, 1)
            self.right_label.opacity = 0.0

            # fade in right label, fade out left label
            anim_duration = Conductor.ticks_per_measure / 4
            self.right_label_alpha = (tick, KFAnim((0.0, 0),
                                                   (ticks_to_next_bar if animated else 0.0, 1.0),
                                                   ((ticks_to_next_bar + anim_duration) if animated else 0.0, 0.0)))
            x_pos = self.right_label.pos[0]
            self.right_label_slide = (tick, KFAnim((0.0, x_pos),
                                                   (ticks_to_next_bar if animated else 0.0, x_pos),
                                                   ((ticks_to_next_bar + anim_duration) if animated else 0.0, Window.width + 100.0),
                                                   ((ticks_to_next_bar + anim_duration + 1) if animated else 0.0, x_pos)))

            self.left_label_alpha = (tick, KFAnim((0.0, 1),
                                                  (ticks_to_next_bar if animated else 0.0, 0.0),
                                                  ((ticks_to_next_bar + anim_duration) if animated else 0.0, 1.0)))
            x_pos = self.left_label.pos[0]
            self.left_label_slide = (tick, KFAnim((0.0, x_pos),
                                                  (ticks_to_next_bar if animated else 0.0, x_pos),
                                                  ((ticks_to_next_bar + 1) if animated else 0.0, -100.0),
                                                  ((ticks_to_next_bar + anim_duration) if animated else 0.0, x_pos)))
            self.scheduler.post_at_tick(self.update_left_label, next_bar + 1)


        if not init:
            self.changing = 1

    def update_left_label(self, tick, ignore):
        self.left_label.color = self.right_label.color
        self.left_label.text = Conductor.harmony_string()

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

        # Label animations
        if self.left_label_alpha:
            start_tick, anim = self.left_label_alpha
            self.left_label.opacity = float(anim.eval(tick - start_tick))

            if not anim.is_active(tick - start_tick):
                self.left_label_alpha = None
        if self.right_label_alpha:
            start_tick, anim = self.right_label_alpha
            self.right_label.opacity = float(anim.eval(tick - start_tick))

            if not anim.is_active(tick - start_tick):
                self.right_label_alpha = None
        # Slides
        if self.left_label_slide:
            start_tick, anim = self.left_label_slide
            self.left_label.pos = (float(anim.eval(tick - start_tick)), self.left_label.pos[1])

            if not anim.is_active(tick - start_tick):
                self.left_label_slide = None
        if self.right_label_slide:
            start_tick, anim = self.right_label_slide
            self.right_label.pos = (float(anim.eval(tick - start_tick)), self.right_label.pos[1])

            if not anim.is_active(tick - start_tick):
                self.right_label_slide = None


        tick_progress = tick-(next_bar-Conductor.ticks_per_measure)

        if self.changing == 1 and self.third_line_anim.is_active(tick_progress):
            self.changing = 2

        if self.changing == 2:
            self.third_color.a = self.change_third_line_anim.eval(tick_progress)

            if not self.change_third_line_anim.is_active(tick_progress):
                self.changing = 0
        else:
            self.third_color.a = self.third_line_anim.eval(tick_progress)
