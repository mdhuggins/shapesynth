from __future__ import print_function
from rtmidi.midiutil import open_midiinput


class Keyboard(object):
    def __init__(self, chord_cb, sustain=True, port=1):
        """ Create a new MIDI keyboard listener. The keyboard keeps track of
            the current chord, and calls chord_cb every time the chord changes.

            If sustain is True, a chord is held until another note outside of
            the chord is played. If sustain is False, the current chord will be
            equivalent to the keys currently held on the keyboard.

            chord_cb should take the set of midi pitches in the current chord
            as its only argument.

        :param chord_cb: callback for every time the chord changes (callable)
        :param sustain: whether to hold a chord until the next one (bool)
        :param port: the MIDI port to use. Defaults to 1 (int)
        """
        # Config
        self.keys_channel = 144  # TODO Is the consistent across controllers?

        # Current chord
        self.chord = set()  # The current chord
        self.chord_cb = chord_cb  # Called every time there's a new chord

        self.held_notes = set()  # The notes currently held on the keyboard

        self.sustain = sustain  # If false, empty chord is possible

        # Setup MIDI
        self.port = port
        Keyboard.set_midi_callback(self.handle_midi, self.port)

    def handle_midi(self, msg, data=None):
        """Callback for handling MIDI input"""
        channel, note, value = msg[0]

        if channel == self.keys_channel:
            if value:
                # Add note
                self.held_notes.add(note)

                # Update chord
                self.chord = self.held_notes.copy()
                self.chord_cb(self.chord)
            else:
                # Remove note
                self.held_notes.remove(note)

                # Check if chord needs to be updated
                if not (self.held_notes < self.chord) or (not self.sustain and not self.held_notes):
                    self.chord = self.held_notes.copy()
                    self.chord_cb(self.chord)

    @staticmethod
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
