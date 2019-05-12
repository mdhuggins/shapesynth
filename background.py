from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Rectangle, Ellipse, Color, BindTexture, Translate, Scale, PushMatrix, PopMatrix, Rotate, RenderContext
from kivy.graphics.stencil_instructions import *
from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.core.image import Image
from kivy.uix.widget import Widget
from common.gfxutil import KFAnim

import time
import numpy as np

fs_multitexture = '''
$HEADER$

// New uniform that will receive texture at index 1
uniform sampler2D texture1;
uniform vec2 translation;
uniform float scale;
uniform float rotation;
uniform vec3 hsvStart;
uniform vec3 hsvEnd;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(void) {

    // multiple current color with both texture (0 and 1).
    // currently, both will use exactly the same texture coordinates.
    vec2 rotated = mat2(cos(rotation), -sin(rotation), sin(rotation), cos(rotation)) * (tex_coord0 - 0.5);

    float t = texture2D(texture1, tex_coord0).a;
    vec3 hsv = t * hsvStart + (1.0 - t) * hsvEnd;

    gl_FragColor = frag_color * \
        vec4(hsv2rgb(hsv), 1.0) * \
        texture2D(texture0, tex_coord0) * \
        texture2D(texture1, (rotated + translation) * scale + 0.5);
}
'''

class Cloud(Widget):

    def __init__(self, pos, size, hue_range=(0.5, 0.7), max_alpha=1.0, **kwargs):
        self.canvas = RenderContext()
        self.canvas.shader.fs = fs_multitexture

        self.reset_parameters(pos, size, hue_range, max_alpha, update=False)

        with self.canvas:
            self.color = Color(1, 1, 1)

            self.translation = Translate(0, 0)
            self.scale = Scale(1)
            self.rotate = Rotate(0)

            BindTexture(source='res/cloud_1_mask.png', index=1)
            Rectangle(pos=(-size[0] / 2, -size[1] / 2), size=size, source='res/cloud_1_noise.png')

        # set the texture1 to use texture index 1
        self.canvas['texture1'] = 1

        # call the constructor of parent
        # if they are any graphics objects, they will be added on our new
        # canvas
        super(Cloud, self).__init__(**kwargs)
        self.on_update(0.0)

    def reset_parameters(self, pos, size, hue_range=(0.5, 0.7), max_alpha=1.0, update=True):
        self.canvas['translation'] = (0, 0)
        self.canvas['scale'] = 1
        self.canvas['rotation'] = 0
        self.canvas['hsvStart'] = (float(hue_range[0]), 0.7, 0.8)
        self.canvas['hsvEnd'] = (float(hue_range[1]), 0.7, 0.8)

        self.mask_translation_rate = np.random.uniform(0.01, 0.5)
        self.mask_scale_rate = np.random.uniform(0.01, 0.5)
        self.mask_rotation_rate = np.random.uniform(-0.2, 0.2) # radians per second
        self.translation_rate = 5.0
        self.scale_rate = np.random.uniform(0.01, 0.5)
        self.rotation_rate = np.random.uniform(-0.1, 0.1) # radians per second
        self.max_alpha = max_alpha

        self.pos = pos
        self.make_animations(pos)
        self.time = 0.0
        if update:
            self.on_update(0.0)


    def make_animations(self, pos):
        """Makes translate, scale, and alpha animations to move the cloud into
        the front."""
        relative_position = (pos[0] - Window.width / 2.0, pos[1] - Window.height / 2.0)
        radius = max(Window.width / 2.0, Window.height / 2.0)
        theta = np.arctan2(relative_position[1], relative_position[0])
        end_point = (np.cos(theta) * radius + Window.width / 2.0, np.sin(theta) * radius + Window.height / 2.0)

        duration = np.random.uniform(10, 20) # todo: make this a parameter
        self.alpha_anim = KFAnim((0.0, 0.0), (duration * 0.2, self.max_alpha), (duration * 0.7, self.max_alpha), (duration * np.random.uniform(0.8, 1.0), 0.0))
        self.translate_anim = KFAnim((0.0, *pos), (duration, *end_point))
        self.scale_anim = KFAnim((0.0, 1.0), (duration, 5.0))

    def on_update(self, dt):
        self.canvas['translation'] = (float(np.cos(self.time * self.mask_translation_rate) * 0.3), float(np.sin(self.time * self.mask_translation_rate) * 0.3))
        self.canvas['scale'] = float(0.5 + np.sin(self.time * self.mask_scale_rate) * 0.03)
        self.canvas['rotation'] += self.mask_rotation_rate * dt
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']
        self.canvas['modelview_mat'] = Window.render_context['modelview_mat']

        self.translation.xy = self.translate_anim.eval(self.time)
        self.scale.x = self.scale_anim.eval(self.time) + np.sin(self.time * self.scale_rate) * 0.03
        self.scale.y = self.scale.x
        self.color.a = self.alpha_anim.eval(self.time)
        self.rotate.angle += self.rotation_rate * dt
        self.time += dt
        return self.alpha_anim.is_active(self.time)


class CloudBackground(Widget):
    creation_interval = 1.0

    def __init__(self, num_clouds, palette, size_range=(200, 800), alpha_range=(0.5, 1), **kwargs):
        super(CloudBackground, self).__init__(**kwargs)
        self.num_clouds = num_clouds
        self.color_palette = palette

        self.clouds = []
        self.margin = Window.width * 0.4
        self.size_range = size_range
        self.alpha_range = alpha_range
        while len(self.clouds) < self.num_clouds:
            self.make_cloud()
        self.last_time = time.time()

    def make_cloud(self, existing_cloud=None):
        pos = (np.random.uniform(self.margin, Window.width - self.margin), np.random.uniform(self.margin, Window.height - self.margin))
        width = np.random.uniform(*self.size_range)
        size = (width, np.random.uniform(width - 100, width + 100))
        if existing_cloud is None:
            new_cloud = Cloud(pos, size, hue_range=self.color_palette.hue_range, max_alpha=np.random.uniform(*self.alpha_range))
            self.add_widget(new_cloud)
            self.clouds.append(new_cloud)
        else:
            existing_cloud.reset_parameters(pos, size, hue_range=self.color_palette.hue_range, max_alpha=np.random.uniform(*self.alpha_range))

    def on_update(self):
        new_time = time.time()
        dt = (new_time - self.last_time) if self.last_time is not None else 0.0
        self.last_time = new_time
        kill_list = [c for c in self.clouds if c.on_update(dt) == False]
        for k in kill_list:
            self.make_cloud(k)
