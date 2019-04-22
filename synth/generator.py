from common.audio import *
import numpy as np


class NoteGenerator(object):
    def __init__(self, pitch, gain, overtones=list([1]), phi=0):
        """ Make a new note generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :param overtones: the relative gains of each overtone, such that
                overtones[0] is the gain of the fundamental frequency,
                overtones[1] is the gain of the 1st overtone, and overtones[n]
                is the gain of the nth overtone. Defaults to [1], which
                corresponds to a pure sine wave at the fundamental frequency.
        """
        super(NoteGenerator, self).__init__()

        # Validate args
        assert type(gain) in (float, int) and gain >= 0
        assert type(pitch) in (float, int)
        assert type(overtones) is list and len(overtones) >= 1 \
               and all([type(o) in (float, int) for o in overtones])

        # Setup generator
        self.freq = self.pitch_to_freq(pitch)
        self.overtones = overtones
        self.gain = gain
        self.phi = phi

        # State information
        self.frame = 0  # Keep angle continuous between generate calls
        self.playing = True

    def note_off(self):
        """ Stop playing.

        :return: None
        """
        self.playing = False

    def generate(self, num_frames, num_channels):
        """ Generate frames.

        :param num_frames: The number of frames to generate (int >= 0)
        :param num_channels: The number of channels to use (must be 1) (int)
        :return: frames, whether the note is playing (np.array([float]), bool)
        """
        # Validate args
        assert num_channels == 1
        assert type(num_frames) is int and num_frames >= 0

        # If not playing, return zeros
        if not self.playing:
            return np.zeros(num_frames), False

        # Generate frames
        output = np.zeros(num_frames)

        for i in range(len(self.overtones)):
            frames = np.arange(self.frame, self.frame + num_frames)
            theta = self.phi + (i+1) * 2 * np.pi * self.freq * frames / Audio.sample_rate
            output += self.overtones[i] * np.sin(theta)

        # Apply gain
        output *= self.gain

        # Save position so next call starts from where this one left off
        self.frame += num_frames
        return output, self.playing

    @staticmethod
    def pitch_to_freq(pitch):
        """ Convert MIDI pitch to frequency.

        :param pitch: MIDI pitch (float)
        :return: frequency, in Hz (float)
        """
        assert type(pitch) in (float, int)
        return 440 * 2**((pitch - 69) / 12)

    @staticmethod
    def sine_wave_generator(pitch, gain):
        """ Make a sine wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a sine wave generator (NoteGenerator)
        """
        return NoteGenerator(pitch, gain)

    @staticmethod
    def square_wave_generator(pitch, gain):
        """ Make a square wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a square wave generator (NoteGenerator)
        """
        overtones = [1 / (k + 1) if k % 2 == 0 else 0 for k in range(30)]
        return NoteGenerator(pitch, gain, overtones)

    @staticmethod
    def triangle_wave_generator(pitch, gain):
        """ Make a triangle wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a triangle wave generator (NoteGenerator)
        """
        # Formula from https://en.wikipedia.org/wiki/Triangle_wave#Harmonics (use sine instead of cosine)
        overtones = [1] + [(((-1.0j)**k) * (k**-2)).real for k in range(1, 30)]
        return NoteGenerator(pitch, gain, overtones)

    @staticmethod
    def sawtooth_wave_generator(pitch, gain):
        """ Make a sawtooth wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a sawtooth wave generator (NoteGenerator)
        """
        overtones = [(-1**(k+2))/(k+1) for k in range(30)]
        return NoteGenerator(pitch, gain, overtones)


class ModulatedGenerator(NoteGenerator):
    def __init__(self, pitch, gain, modulator, overtones=list([1]), mod_gain=5):
        """ Make a new note generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :param overtones: the relative gains of each overtone, such that
                overtones[0] is the gain of the fundamental frequency,
                overtones[1] is the gain of the 1st overtone, and overtones[n]
                is the gain of the nth overtone. Defaults to [1], which
                corresponds to a pure sine wave at the fundamental frequency.
        """
        super(ModulatedGenerator, self).__init__(pitch, gain, overtones)

        self.modulator = modulator
        self.mod_gain = 10**mod_gain  # In decibels  # TODO Figure this out

    def generate(self, num_frames, num_channels):
        """ Generate frames.

        :param num_frames: The number of frames to generate (int >= 0)
        :param num_channels: The number of channels to use (must be 1) (int)
        :return: frames, whether the note is playing (np.array([float]), bool)
        """
        # Validate args
        assert num_channels == 1
        assert type(num_frames) is int and num_frames >= 0

        # If not playing, return zeros
        if not self.playing:
            return np.zeros(num_frames), False

        # Generate frames
        mod_frames = self.modulator.generate(num_frames, num_channels)[0]
        mod_omega = 2*np.pi*self.modulator.freq

        output = np.zeros(num_frames)

        for i in range(len(self.overtones)):
            frames = np.arange(self.frame, self.frame + num_frames)
            theta = (i+1) * 2 * np.pi * self.freq * frames / Audio.sample_rate
            output += self.overtones[i] * np.sin(theta + self.mod_gain*mod_frames/mod_omega)

        # Apply gain
        output *= self.gain

        # Save position so next call starts from where this one left off
        self.frame += num_frames

        return output, self.playing

    @staticmethod
    def sine_wave_generator(pitch, gain, modulator):
        """ Make a sine wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a sine wave generator (NoteGenerator)
        """
        return ModulatedGenerator(pitch, gain, modulator)

    @staticmethod
    def square_wave_generator(pitch, gain, modulator):
        """ Make a square wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a square wave generator (NoteGenerator)
        """
        overtones = [1 / (k + 1) if k % 2 == 0 else 0 for k in range(30)]
        return ModulatedGenerator(pitch, gain, modulator, overtones)

    @staticmethod
    def triangle_wave_generator(pitch, gain, modulator):
        """ Make a triangle wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a triangle wave generator (NoteGenerator)
        """
        # Formula from https://en.wikipedia.org/wiki/Triangle_wave#Harmonics (use sine instead of cosine)
        overtones = [1] + [(((-1.0j)**k) * (k**-2)).real for k in range(1, 30)]
        return ModulatedGenerator(pitch, gain, modulator, overtones)

    @staticmethod
    def sawtooth_wave_generator(pitch, gain, modulator):
        """ Make a sawtooth wave generator.

        :param pitch: the MIDI pitch to generate (int)
        :param gain: the amplitude of the output (float >=0)
        :return: a sawtooth wave generator (NoteGenerator)
        """
        overtones = [(-1**(k+2))/(k+1) for k in range(30)]
        return ModulatedGenerator(pitch, gain, modulator, overtones)
