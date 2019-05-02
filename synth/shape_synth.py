
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

        self.on_note = None

    def make_note(self, pitch, velocity, duration):
        """ Creates a generator to play a note.

        :param pitch: MIDI
        :param velocity: in range [0,1]
        :param duration: in seconds
        :return: a generator
        """
        if self.on_note is not None:
            self.on_note(pitch, velocity, duration)

        # x = self.x
        # y = self.y

        gain = self.gain * velocity

        # mixer = Mixer()
        #
        # # Carrier Params
        # carrier_p = max(0, 1-y-(1-x)/4)
        # carrier_env_params = Envelope.magic_envelope(carrier_p, duration=duration)
        # # carrier_gain = gain * ((y) * (1 - x)) ** (1 / 2.5)
        # carrier_gain = gain * ((y) * (1 - x)) ** (1 / 2.5)
        # if y < 0.25:
        #     carrier_gain = (1-x)*((1-y)**2)*0.5
        #
        # # Modulator Params
        # modulator_env_params = carrier_env_params
        #
        # # FM Factory
        # # fm_fact = FMFactory(carrier_gain, 0, 1, 1, carrier_env_params, modulator_env_params)
        # # mixer.add(fm_fact.create_fm(pitch))
        #
        # # Roughness
        # tri_gain = max(0,1-self.roughness)*carrier_gain
        # square_gain = min(1, self.roughness)*carrier_gain
        #
        # triangle = NoteGenerator.triangle_wave_generator(pitch, tri_gain)
        # square = NoteGenerator.sawtooth_wave_generator(pitch, square_gain)
        #
        # note_env_params = Envelope.magic_envelope(carrier_p)
        # triangle = Envelope(triangle, *note_env_params)
        # square = Envelope(square, *note_env_params)
        #
        # mixer.add(triangle)
        # mixer.add(square)
        #
        # # Noise Params
        # # noise_p = max(0, 1 - x**2/3.5 - (1-y)*x/15)
        # noise_p = 1-x
        # noise_env_params = Envelope.magic_envelope(noise_p)
        # noise_gain = 0#gain * 1 * (1-y)**2  #*0.2
        #
        #
        # # Noise Generator
        # noise = NoiseGenerator(noise_gain)
        # noise = Envelope(noise, *noise_env_params)
        #
        # # Noise Filter
        # f0 = pitch_to_freq(pitch)
        # # noise_cutoffs = [0.1 * f0, min(1.9 * f0 ** 1.3, Audio.sample_rate / 2 - 1)]
        # noise_cutoffs = [f0] if x < 0.5 else [min(f0 ** 1.3, Audio.sample_rate / 2 - 1)]
        # filter_type = 'lowpass' if x < 0.5 else 'highpass'
        # noise = Filter(noise, filter_type, noise_cutoffs)
        #
        # mixer.add(noise)

        return DaemonClientGenerator(SamplerManager.pool, (self.gain, (self.x, self.y)), pitch, gain)#, hash)
