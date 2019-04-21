
# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.synth import *
from common.clock import *

import numpy as np

from synth import *

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


class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        self.audio = Audio(1)  # TODO Stereo

        p = 0.1
        time.sleep(1)

        self.mod = NoteGenerator.square_wave_generator(84, 0.05)
        self.mod = Envelope.magic_envelope(self.mod, p)
        self.note_gen = ModulatedGenerator.sine_wave_generator(60, 0.3, self.mod)

        self.env = Envelope.magic_envelope(self.note_gen, p)
        self.audio.set_generator(self.env)

    def on_update(self):
        self.audio.on_update()


run(MainWidget, title="Synth Test")
