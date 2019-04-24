from __future__ import print_function
from rtmidi.midiutil import open_midiinput


def set_midi_callback(callback, port=1):
    """ Starts listening for MIDI input on a port. If that port is not
        available, prompts for choice on console. Calls callback every
        time MIDI input is received.

        Callbacks should take two arguments. The first is a tuple of a list of
        integers [channel, note, value], and the time, in seconds, since the
        last input. The tuple looks like this: ([channel, note, value], dt)

        The second argument is an optional data field.

    :param callback: called every time there's a MIDI input (callable)
    :param port: the MIDI port to listen on. Defaults to 1. (int)
    :return: None
    """
    midiin, port_name = open_midiinput(port)
    midiin.set_callback(callback)
