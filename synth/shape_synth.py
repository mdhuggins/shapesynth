
# common import
import sys
sys.path.append('..')

from common.mixer import *

from .noise import *
from .filter import *
from .fm import *
from .envelope import *
from .util import pitch_to_freq
from .sampler import *

# TODO Use shape in sound
# TODO Translate updates sound

class DaemonClientGenerator(object):
    """
    Requests the sampler daemon to produce frames, then plays the resulting
    sound when generate() is called. If the daemon has not yet returned the
    sound, skips frames until the daemon returns.
    """

    def __init__(self, pool, *params):
        self.pool = pool
        self.pool.request(id(self), params)
        self.frames = None
        self.frame = 0

    def generate(self, num_frames, num_channels):
        if self.frames is None:
            self.frames = self.pool.get(id(self))

        if self.frames is None:
            print("Can't play yet - no frames loaded!")
            return np.zeros(num_frames * num_channels), True

        output = self.frames[self.frame:self.frame+num_frames]
        actual_num_frames = len(output) // num_channels
        self.frame += actual_num_frames

        # Pad if output is too short
        padding = num_frames * num_channels - len(output)
        if padding > 0:
            output = np.append(output, np.zeros(padding))

        return output, self.frame < len(self.frames)

class ShapeSynth(object):
    def __init__(self, x, y, gain, roughness):
        """ Create a new ShapeSynth.

        :param x: the relative x coordinate of the shape (float in range [0,1])
        :param y: the relative y coordinate of the shape (float in range [0,1])
        :param gain: in range [0,1]
        """
        super(ShapeSynth, self).__init__()

        # TODO update property
        self.x = x
        self.y = y
        self.gain = gain

        self.roughness = roughness

    def make_note(self, pitch, velocity, duration):
        """ Creates a generator to play a note.

        :param pitch: MIDI
        :param velocity: in range [0,1]
        :param duration: in seconds
        :return: a generator
        """

        return DaemonClientGenerator(SamplerManager.pool, (self.gain, (self.x, self.y)), pitch, self.gain * velocity)
