
# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.synth import *
from common.clock import *
from common.mixer import *

import numpy as np

from synth import *
from noise import *
from filter import *
import time

# if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    #
    # note_gen = NoteGenerator.sine_wave_generator(60, 0.5)
    #
    # for p in range(11):
    #     pp = p/10
    #     env = Envelope(note_gen, pp)
    #
    #     plt.figure(str(pp))
    #     plt.xlim(0, 60000)
    #     plt.plot(env.envelope)
    #     plt.show()


from fm import *
from shape_synth import *

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        self.audio = Audio(1)  # TODO Stereo

        time.sleep(1)

        # self.mod = NoteGenerator.square_wave_generator(84, 0.05)
        # self.mod = Envelope.magic_envelope(self.mod, p)
        # self.note_gen = ModulatedGenerator.sine_wave_generator(60, 0.3, self.mod)
        #
        # self.env = Envelope.magic_envelope(self.note_gen, p)

        car_ps = Envelope.magic_envelope(1)
        mod_ps = Envelope.magic_envelope(1)

        fm_fact = FMFactory(0.5, 0, 1, 1, car_ps, mod_ps)

        # gen = fm_fact.create_fm(48)

        self.mixer = Mixer()

        self.audio.set_generator(self.mixer)


    def on_update(self):
        self.audio.on_update()

    def on_touch_down(self, touch):
        x, y = touch.pos

        xx = x / (2*self.center_x)
        yy = y / (2*self.center_y)
        pitch = int(66 + (18 + 36 * (1 - yy)) * (xx - 0.5))

        ss = ShapeSynth(xx, yy)

        self.mixer.add(ss.make_note(pitch, 1))



run(MainWidget, title="Synth Test")
