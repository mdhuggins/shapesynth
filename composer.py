import numpy as np
from common.clock import *

class Conductor(object):
    """
    Provides evolving parameters for the composers.
    """
    triplet_preference = 0.5
    speed_preference = 0.5
    obedience_preference = 1.0
    scheduler = None
    harmony = [0, 4, 7]
    scale = [0, 2, 4, 5, 7, 9, 11]
    playing = False

    @staticmethod
    def initialize(sched):
        Conductor.scheduler = sched

    @classmethod
    def start(self):
        if self.playing: return
        self.playing = True
        next_beat = quantize_tick_up(self.scheduler.get_tick(), kTicksPerQuarter)
        self.update_cmd = self.scheduler.post_at_tick(Conductor.update, next_beat)

    @classmethod
    def stop(self):
        if not self.playing: return
        self.playing = False
        self.sched.remove(self.update_cmd)

    @classmethod
    def toggle(self):
        if self.playing: self.stop()
        else: self.start()

    @classmethod
    def update(self, tick, ignore):
        change_flag = np.random.random()
        if change_flag < 0.3:
            Conductor.triplet_preference = np.clip(Conductor.triplet_preference + 0.1, 0.0, 1.0)
        elif change_flag < 0.6:
            Conductor.triplet_preference = np.clip(Conductor.triplet_preference + 0.1, 0.0, 1.0)

        change_flag = np.random.random()
        if change_flag < 0.3:
            Conductor.speed_preference = np.clip(Conductor.speed_preference + 0.1, 0.0, 1.0)
        elif change_flag < 0.6:
            Conductor.speed_preference = np.clip(Conductor.speed_preference - 0.1, 0.0, 1.0)

        change_flag = np.random.random()
        if change_flag < 0.3:
            Conductor.obedience_preference = np.clip(Conductor.obedience_preference + 0.1, 0.0, 1.0)
        elif change_flag < 0.6:
            Conductor.obedience_preference = np.clip(Conductor.obedience_preference - 0.1, 0.0, 1.0)

        next_beat = quantize_tick_up(self.scheduler.get_tick(), kTicksPerQuarter)
        self.update_cmd = self.scheduler.post_at_tick(Conductor.update, next_beat)

RHYTHMS = [
    # base probability, triplet, speed, complexity, rhythm (sums to 480)
    (8, 0.0, 0.0, 0.05, [480]),
    (10, 0.0, 0.0, 0.1, [240, 240]),
    (10, 0.0, 0.0, 0.15, [240, 120, 120]),
    (10, 0.0, 0.0, 0.15, [120, 120, 240]),

    (5, 0.0, 0.25, 0.15, [120, 120, 120, 120]),
    (5, 0.0, 0.25, 0.2, [60, 60, 120, 120, 120]),
    (5, 0.0, 0.25, 0.3, [120, 60, 60, 120, 120]),
    (5, 0.0, 0.25, 0.25, [120, 120, 60, 60, 120]),
    (5, 0.0, 0.25, 0.3, [120, 120, 120, 60, 60]),
    (5, 0.0, 0.25, 0.5, [120, 60, 120, 60, 60, 60]),
    (5, 0.0, 0.25, 0.5, [120, 60, 120, 120, 60]),
    (5, 0.0, 0.25, 0.5, [60, 120, 120, 60, 120]),

    (3, 0.0, 0.6, 0.3, [60, 60, 120, 60, 60, 120]),
    (3, 0.0, 0.6, 0.35, [60, 60, 120, 120, 60, 60]),
    (3, 0.0, 0.6, 0.3, [120, 60, 60, 120, 60, 60]),
    (3, 0.0, 0.6, 0.4, [120, 60, 60, 60, 60, 120]),
    (3, 0.0, 0.6, 0.35, [120, 120, 60, 60, 60, 60]),
    (3, 0.0, 0.6, 0.35, [120, 120, 60, 60, 60, 60]),

    (1, 0.25, 0.4, 0.4, [120, 40, 40, 40, 120, 120]),
    (1, 0.25, 0.4, 0.4, [120, 120, 40, 40, 40, 120]),
    (1, 0.5, 0.4, 0.5, [40, 40, 40, 120, 40, 40, 40, 120]),
    (1, 0.5, 0.4, 0.5, [120, 120, 40, 40, 40, 40, 40, 40]),
    (1, 1.0, 0.4, 0.6, [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]),
]
MAX_PITCH = 96

class Composer(object):
    """
    Generates music for a single instrument.
    """

    def __init__(self, sched, mixer, note_factory, pitch_level, pitch_variance, complexity, harmonic_obedience, update_interval):
        """
        Initializes a Composer that uses the given note factory to create note
        generators.

        sched: a scheduler on which notes will be queued
        note_factory: a callable that takes a MIDI pitch, velocity in [0.0, 1.0],
            and duration in seconds, and returns a note generator
        pitch_level: a float in [0.0, 1.0] that indicates how low/high the pitches
            should be
        pitch_variance: a float in [0.0, 1.0] indicating how much to vary the
            pitch
        complexity: a float in [0.0, 1.0] indicating how fast and varied the
            rhythm should be
        harmonic_obedience: a float in [0.0, 1.0] indicating how strongly the
            pitches should follow the harmony
        update_interval: the number of beats' worth of music that should be
            generated at a time
        """
        self.sched = sched
        self.mixer = mixer
        self.note_factory = note_factory
        self.pitch_level = pitch_level
        self.pitch_variance = pitch_variance
        self.complexity = complexity
        self.harmonic_obedience = harmonic_obedience
        self.update_interval = update_interval
        self.playing = False
        self.queued_notes = [] # Will contain lists of commands
        self.rhythm_indexes = []

    def start(self):
        if self.playing: return
        self.playing = True
        def update(tick, ignore):
            next_beat = quantize_tick_up(tick + 1, self.update_interval * kTicksPerQuarter)
            self.update_composition(next_beat)
            self.update_cmd = self.sched.post_at_tick(update, next_beat + self.update_interval * kTicksPerQuarter - np.random.randint(100, 200))
        self.update_cmd = self.sched.post_at_tick(update, self.sched.get_tick())

    def stop(self):
        if not self.playing: return
        self.playing = False
        self.sched.remove(self.update_cmd)

    def toggle(self):
        if self.playing: self.stop()
        else: self.start()

    def update_composition(self, next_beat):
        """
        Creates an `update_interval` worth of composition that will start at
        `next_beat`.
        """
        # Get the last note if available
        if len(self.queued_notes) > 0:
            current_measure = self.queued_notes.pop(0)
            if len(current_measure) > 0:
                last_note = current_measure[-1][0]
        else:
            last_note = None
        if np.random.random() < self.pitch_variance:
            last_note = None

        if len(self.rhythm_indexes) > 0:
            last_rhythm = self.rhythm_indexes.pop(0)
        else:
            last_rhythm = None

        rhythm, new_notes = self.generate_note_sequence(last_note, last_rhythm)
        self.queued_notes.append(new_notes)
        self.rhythm_indexes.append(rhythm)

        # Schedule the notes to be played
        current_tick = 0
        for note_params in new_notes:
            self.sched.post_at_tick(self.play_note, next_beat + current_tick, note_params)
            current_tick += note_params[1]

    def play_note(self, tick, note_params):
        """
        Called by the scheduler to create a note with the given parameters.
        """
        pitch, dur = note_params
        dur_sec = self.sched.tempo_map.tick_to_time(tick + dur) - self.sched.tempo_map.tick_to_time(tick)
        # Currently not using note velocity
        note_gen = self.note_factory(pitch, 1.0, dur_sec)
        self.mixer.add(note_gen)

    def generate_note_sequence(self, last_note=None, last_rhythm=None):
        """
        Selects a rhythm and pitches, using the given last note if available.
        """

        new_notes = []
        index, rhythm = self.pick_rhythm(last_rhythm=last_rhythm)
        print(rhythm)
        current_tick = 0
        for duration in rhythm:
            last_note = self.pick_note(current_tick, last_note=last_note)
            new_notes.append((last_note, duration))
            current_tick += duration
        return index, new_notes

    def pick_rhythm(self, last_rhythm=None):
        """
        Selects a sequence of durations to add up to `update_interval` beats.
        last_rhythm may be an index of a rhythm previously used.
        """
        probs = []
        for index, (baseline, triplet, speed, complexity, rhythm) in enumerate(RHYTHMS):
            prob = baseline
            prob += self.rhythm_property_factor(triplet, Conductor.triplet_preference)
            prob += self.rhythm_property_factor(speed, Conductor.speed_preference)
            prob += self.rhythm_property_factor(complexity, self.complexity)
            if last_rhythm is not None:
                distance = abs(last_rhythm - index)
                prob *= np.exp(-(max(distance, 3) - 3) ** 2 / (2 * 10 ** max(self.complexity, 1e-6)))
            probs.append(prob)

        probs = np.array(probs)
        probs /= np.sum(probs)
        print(last_rhythm, probs)
        rhythm_index = np.random.choice(range(len(RHYTHMS)), p=probs)
        base_rhythm = np.array(RHYTHMS[rhythm_index][-1])
        return rhythm_index, (base_rhythm * self.update_interval).tolist()

    def rhythm_property_factor(self, factor, preferred):
        """
        Computes a scale value determining how likely to choose a rhythm
        with the given property `factor`, given that the composer prefers rhythms
        at the `preferred` amount of that property.
        """

        steepness = 60.0
        result = (1.0 - preferred) ** 3 / (factor + (1.0 / steepness))
        result += preferred ** 3 / (1 - factor + (1.0 / steepness))
        return result

    def obedience_factor(self, tick):
        """
        Computes a harmonic obedience given a tick value, such that strong
        beats are more harmonically obedient.
        """

        obed = np.clip((self.harmonic_obedience + Conductor.obedience_preference) / 2.0, 0.0, 1.0)
        for i, tick_mod in [(4, 480), (8, 480 * 2), (16, 480 * 4)]:
            if tick % tick_mod == 0 and tick != 0:
                obed = obed / i + (i - 1) / i
        return obed

    def pick_note(self, tick, last_note=None):
        """
        Selects a pitch for the next note to play, based on the given last note
        (or None). Uses the given tick value to dilate the harmonic obedience
        so that notes on strong beats are preferentially
        """
        obedience_factor = self.obedience_factor(tick)

        probs = {}
        for pitch_class in Conductor.scale:
            probs[pitch_class] = 1.0

        # Make pitch classes in harmony more likely
        for pitch_class in Conductor.harmony:
            probs[pitch_class] = probs.get(pitch_class, 1.0) * 1 / (1 - obedience_factor)

        if last_note is not None:
            last_pitch_class = last_note % 12
            # Weight pitches closer to this pitch class
            for pitch_class in probs:
                distance = min((pitch_class - last_pitch_class) % 12, (last_pitch_class - pitch_class) % 12)
                probs[pitch_class] *= np.exp(-(max(distance, 2) - 1) ** 2 / (2 * 10 ** max(self.pitch_variance, 1e-6)))

        pitch_list = sorted(probs.keys())
        pitch_weights = np.array([probs[k] for k in pitch_list])
        pitch_weights /= np.sum(pitch_weights)
        selected_pitch_class = np.random.choice(pitch_list, p=pitch_weights)

        if last_note is None:
            pitch = selected_pitch_class + 12 * int(self.pitch_level * 9)
        else:
            up_pitch = last_note + (selected_pitch_class - last_pitch_class) % 12
            down_pitch = last_note - (last_pitch_class - selected_pitch_class) % 12
            pitch = min([up_pitch, down_pitch], key=lambda p: abs(p - last_note))

        # Don't let pitch get too high
        while pitch > MAX_PITCH:
            pitch -= 12

        return pitch

if __name__ == '__main__':
    composer = Composer(None, None, None, 0.5, 0.001, 0.5, 0.1, 4)
    for i in range(10):
        print(composer.generate_note_sequence())
