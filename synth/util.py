def pitch_to_freq(pitch):
    """ Convert MIDI pitch to frequency.

    :param pitch: MIDI pitch (float)
    :return: frequency, in Hz (float)
    """
    return 440 * 2 ** ((pitch - 69) / 12)