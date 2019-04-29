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
    harmony = [0, 4, 7, 9]
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
        self.scheduler.remove(self.update_cmd)

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

        next_beat = quantize_tick_up(self.scheduler.get_tick() + 1, kTicksPerQuarter * 8)
        self.update_cmd = self.scheduler.post_at_tick(Conductor.update, next_beat)

RHYTHMS = [
    # base probability, triplet, speed, complexity, rhythm (sums to 480)
    (8, 0.0, 0.0, 0.05, [480]),
    (10, 0.0, 0.0, 0.1, [240, 240]),
    (9, 0.0, 0.0, 0.1, [240, -240]),
    (5, 0.0, 0.0, 0.1, [-240, 240]),
    (6, 0.0, 0.0, 0.15, [-240, 120, 120]),
    (10, 0.0, 0.0, 0.15, [240, 120, 120]),
    (6, 0.0, 0.0, 0.15, [240, 120, -120]),
    (10, 0.0, 0.0, 0.15, [120, 120, 240]),

    (5, 0.0, 0.25, 0.15, [120, 120, 120, 120]),
    (5, 0.0, 0.25, 0.2, [60, 60, 120, 120, 120]),
    (5, 0.0, 0.25, 0.2, [60, 60, 120, -120, 120]),
    (5, 0.0, 0.25, 0.3, [120, 60, 60, 120, 120]),
    (5, 0.0, 0.25, 0.3, [120, 60, 60, 120, -120]),
    (4, 0.0, 0.25, 0.25, [120, 120, 60, 60, 120]),
    (4, 0.0, 0.25, 0.3, [120, 120, 120, 60, 60]),
    (4, 0.0, 0.25, 0.5, [120, 60, 120, 60, 60, 60]),
    (4, 0.0, 0.25, 0.5, [120, 60, 120, 120, 60]),
    (4, 0.0, 0.25, 0.5, [60, 120, 120, 60, 120]),

    (3, 0.0, 0.6, 0.3, [60, 60, 120, 60, 60, 120]),
    (3, 0.0, 0.6, 0.35, [60, 60, 120, 120, 60, 60]),
    (3, 0.0, 0.6, 0.3, [120, 60, 60, 120, 60, 60]),
    (3, 0.0, 0.6, 0.4, [120, 60, 60, 60, 60, 120]),
    (3, 0.0, 0.6, 0.35, [120, 120, 60, 60, 60, 60]),
    (3, 0.0, 0.6, 0.35, [120, 120, 60, 60, 60, 60]),
    (3, 0.0, 0.6, 0.35, [120, -120, 60, 60, 60, 60]),

    (1, 0.25, 0.4, 0.4, [120, 180, 60, 120, 120]),
    (1, 0.25, 0.4, 0.4, [120, 120, 180, 60, 120]),
    (1, 0.5, 0.4, 0.5, [180, 60, 120, 180, 60, 120]),
    (1, 0.5, 0.4, 0.5, [180, 180, 180, 180, 60, -60]),

    (1, 0.1, 0.8, 0.7, [30, 30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 60]),
    (1, 0.1, 0.8, 0.8, [60, 60, 60, 60, 30, 30, 30, 30, 30, 30, 30, 30]),
    (1, 0.1, 0.8, 0.9, [30, 30, 30, 30, 60, 30, 30, 30, 30, 30, 30, 60, 30, 30]),
    (1, 0.1, 0.8, 0.8, [60, 30, 30, 30, 30, 60, 90, 30, 60, 60]),
    #(1, 0.5, 0.4, 0.5, [120, 120, 160, 40, 40, 40, 40]),

    (1, 0.25, 0.4, 0.7, [120, 40, 40, 40, 120, 120]),
    (1, 0.25, 0.4, 0.7, [120, 120, 40, 40, 40, 120]),
    (1, 0.5, 0.4, 0.8, [40, 40, 40, 120, 40, 40, 40, 120]),
]
MAX_PITCH = 96

class Composer(object):
    """
    Generates music for a single instrument.
    """

    def __init__(self, sched, mixer, note_factory, pitch_level=0.0, pitch_variance=0.0, velocity_level=0.0, velocity_variance=0.0, complexity=0.0, harmonic_obedience=0.0, bass_preference=0.0, arpeggio_preference=0.5, update_interval=4):
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
        velocity_level: a float in [0.0, 1.0] indicating the average velocity
            of notes
        velocity_variance: a float in [0.0, 1.0] indicating how much to vary
            the velocity
        complexity: a float in [0.0, 1.0] indicating how fast and varied the
            rhythm should be
        harmonic_obedience: a float in [0.0, 1.0] indicating how strongly the
            pitches should follow the harmony
        bass_preference: a float in [0.0, 1.0] indicating how much to prefer
            the bass note of the harmony
        update_interval: the number of beats' worth of music that should be
            generated at a time
        """
        self.sched = sched
        self.mixer = mixer
        self.note_factory = note_factory
        self.pitch_level = pitch_level
        self.pitch_variance = pitch_variance
        self.velocity_level = velocity_level
        self.velocity_variance = velocity_variance
        self.complexity = complexity
        self.harmonic_obedience = harmonic_obedience
        self.bass_preference = bass_preference
        self.arpeggio_preference = arpeggio_preference
        self.update_interval = update_interval
        self.playing = False
        self.measure_stack = [] # Will contain lists of commands
        self.queued_measures = [] # Measures to play in the future
        self.last_rhythm = None
        self.scheduled_to_beat = None

    def start(self):
        if self.playing: return
        self.playing = True

        self._update(self.sched.get_tick(), None)

    def stop(self):
        if not self.playing: return
        self.playing = False
        self.scheduled_to_beat = None
        self.sched.remove(self.update_cmd)

    def toggle(self):
        if self.playing: self.stop()
        else: self.start()

    def clear_notes(self):
        self.measure_stack = []
        self.queued_measures = []

    def _update(self, tick, ignore):
        """Helper method called periodically by scheduler to start asynchronous processing."""
        if self.scheduled_to_beat is not None and tick < self.scheduled_to_beat - self.update_interval * kTicksPerQuarter:
            return

        next_beat = quantize_tick_up(tick + kTicksPerQuarter, self.update_interval * kTicksPerQuarter)
        new_notes = self.update_composition(next_beat)
        if new_notes is not None:
            current_tick = 0
            for note_params in new_notes:
                if note_params[0] is not None:
                    self.sched.post_at_tick(self.play_note, next_beat + current_tick, note_params)
                current_tick += note_params[2]

            # Schedule the next update
            self.scheduled_to_beat = next_beat + min(current_tick, kTicksPerQuarter * self.update_interval)
            time = self.scheduled_to_beat - np.random.randint(50, int(self.update_interval * 0.3 * kTicksPerQuarter))
            self.update_cmd = self.sched.post_at_tick(self._update, time)

    def update_composition(self, next_beat):
        """
        Creates at least `update_interval` worth of composition that will start at
        `next_beat`. Returns the number of ticks that this segment of composition
        will last.
        """
        new_sequence = None

        # Play the next queued sequence if available
        if len(self.queued_measures) > 0:
            new_sequence = self.queued_measures.pop(0)

        # Possibly pop a random amount of past sequences off the stack
        if new_sequence is None and len(self.measure_stack) > 0:
            pop_level = 2 ** int(np.log2(len(self.measure_stack)))
            while pop_level >= 1 and new_sequence is None:
                if np.random.random() < 0.5 * (1 - self.complexity):
                    self.queued_measures = self.measure_stack[-pop_level:]
                    new_sequence = self.queued_measures.pop(0)
                    del self.measure_stack[-pop_level:]
                    break
                pop_level = pop_level // 2

        # Generate a new sequence
        if new_sequence is None:
            # Get the last note if available
            if len(self.measure_stack) > 0:
                current_measure = self.measure_stack[-1]
                if len(current_measure) > 0:
                    last_note = current_measure[-1]
            else:
                last_note = None
            # Maybe ignore the last note
            if np.random.random() < self.pitch_variance:
                last_note = None

            rhythm, new_sequence = self.generate_note_sequence(last_note, self.last_rhythm)
            self.last_rhythm = rhythm

        if new_sequence is not None:
            self.measure_stack.append(new_sequence)

        return new_sequence

    def play_note(self, tick, note_params):
        """
        Called by the scheduler to create a note with the given parameters.
        """
        pitch, velocity, dur = note_params
        dur_sec = self.sched.tempo_map.tick_to_time(tick + dur) - self.sched.tempo_map.tick_to_time(tick)
        # Currently not using note velocity
        note_gen = self.note_factory(pitch, velocity, dur_sec)
        self.mixer.add(note_gen)

    def generate_note_sequence(self, last_note=None, last_rhythm=None):
        """
        Selects a rhythm and pitches, using the given last note if available.
        """

        new_notes = []
        index, rhythm = self.pick_rhythm(last_rhythm=last_rhythm)
        print(rhythm)
        current_tick = 0
        if np.random.random() < self.arpeggio_preference:
            new_notes = self.make_arpeggio(rhythm, last_note=last_note)
        else:
            for duration in rhythm:
                if duration < 0:
                    new_notes.append((None, None, -duration))
                    current_tick += -duration
                else:
                    last_note = self.pick_note(current_tick, duration, last_note=last_note)
                    new_notes.append(last_note)
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
        rhythm_index = np.random.choice(range(len(RHYTHMS)), p=probs)
        base_rhythm = np.array(RHYTHMS[rhythm_index][-1])
        return rhythm_index, (base_rhythm * self.update_interval).tolist()

    def rhythm_property_factor(self, factor, preferred):
        """
        Computes a scale value determining how likely to choose a rhythm
        with the given property `factor`, given that the composer prefers rhythms
        at the `preferred` amount of that property.
        """

        steepness = 80.0
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

    def pick_note(self, tick, duration, last_note=None):
        """
        Selects a pitch for the next note to play, based on the given last note,
        which contains the pitch, velocity, and duration (or None). Uses the
        given tick value to dilate the harmonic obedience so that notes on strong
        beats are preferentially harmonic.
        """
        obedience_factor = self.obedience_factor(tick)

        probs = {}
        for pitch_class in Conductor.scale:
            probs[pitch_class] = 1.0

        # Make pitch classes in harmony more likely
        for pitch_class in Conductor.harmony:
            probs[pitch_class] = probs.get(pitch_class, 1.0) * 5 / (1 - obedience_factor) ** 2
        probs[Conductor.harmony[0]] *= 1 / (1 - self.bass_preference)

        if last_note is not None and last_note[0] is not None:
            last_pitch_class = last_note[0] % 12
            # Weight pitches closer to this pitch class
            for pitch_class in probs:
                distance = min((pitch_class - last_pitch_class) % 12, (last_pitch_class - pitch_class) % 12)
                probs[pitch_class] *= np.exp(-(max(distance, 2) - 2) ** 2 / (2 * 10 ** max(self.pitch_variance, 1e-6)))

        pitch_list = sorted(probs.keys())
        pitch_weights = np.log(1 + np.array([probs[k] for k in pitch_list]))
        pitch_weights /= np.sum(pitch_weights)
        selected_pitch_class = np.random.choice(pitch_list, p=pitch_weights)

        # Choose final pitch
        if last_note is None or last_note[0] is None:
            pitch = selected_pitch_class + 12 * int(self.pitch_level * 9)
        else:
            up_pitch = last_note[0] + (selected_pitch_class - last_pitch_class) % 12
            down_pitch = last_note[0] - (last_pitch_class - selected_pitch_class) % 12
            pitch = min([up_pitch, down_pitch], key=lambda p: abs(p - last_note[0]))

        # Don't let pitch get too high
        while pitch > MAX_PITCH:
            pitch -= 12

        # Choose velocity
        if last_note is None or last_note[1] is None:
            velocity = np.random.normal(self.velocity_level, self.velocity_variance)
        else:
            last_velocity = last_note[1]
            mean = (pitch - last_note[0]) / 20.0
            velocity = last_velocity + np.random.normal(mean, self.velocity_variance ** 2)
        velocity = np.clip(velocity, 0.1, 1.0)

        return pitch, velocity, duration

    def make_arpeggio(self, durations, last_note=None):
        """
        Builds an arpeggio on the notes in the conductor's harmony, for a rhythm
        with the given durations. Returns a list of [(pitch, velocity, duration)].
        """
        notes = []
        if len(durations) == 0: return notes

        if last_note is not None and last_note[0] is not None:
            # Find closest note in the harmony to last_note
            last_pitch = last_note[0]
            last_pitch_class = last_pitch % 12
            closest_pitch = min(Conductor.harmony, key=lambda p: min(abs((p - last_pitch_class) % 12), abs((last_pitch_class - p) % 12)))

        # If closest_pitch is the same as last pitch, don't use it automatically
        if last_note is None or last_note[0] is None or last_pitch_class == closest_pitch:
            closest_pitch = np.random.choice(Conductor.harmony)

        # Pick octave
        if last_note is not None and last_note[0] is not None:
            up_pitch = last_pitch + (closest_pitch - last_pitch_class) % 12
            down_pitch = last_pitch - (last_pitch_class - closest_pitch) % 12
            pitch = min([up_pitch, down_pitch], key=lambda p: abs(p - last_pitch))
        else:
            pitch = closest_pitch + 12 * int(self.pitch_level * 9)

        velocity = last_note[1] if last_note is not None and last_note[1] is not None else np.clip(np.random.normal(self.velocity_level, self.velocity_variance), 0.1, 1.0)
        notes.append((pitch, velocity, durations[0]))

        # Pick the remainder of the notes by steps in the harmony
        pitch_index = Conductor.harmony.index(int(closest_pitch))
        while len(notes) < len(durations):
            last_pitch_class = Conductor.harmony[pitch_index]
            if np.random.random() < 0.5: # Going up
                probs = np.array([0.01, 0.01, 0.03, 0.8, 0.15])
            else:                        # Going down
                probs = np.array([0.15, 0.8, 0.03, 0.01, 0.01])
            pitch_index = int(pitch_index + np.random.choice(np.arange(-2, 3), p=probs)) % len(Conductor.harmony)
            new_pitch_class = Conductor.harmony[pitch_index]

            # Choose octave
            up_pitch = notes[-1][0] + (new_pitch_class - last_pitch_class) % 12
            down_pitch = notes[-1][0] - (last_pitch_class - new_pitch_class) % 12
            pitch = min([up_pitch, down_pitch], key=lambda p: abs(p - notes[-1][0]))
            while pitch - notes[-1][0] > 12:
                pitch -= 12
            while pitch - notes[-1][0] < -12:
                pitch += 12

            # Choose velocity
            mean = (pitch - notes[-1][0]) / 20.0
            velocity = np.clip(notes[-1][1] + np.random.normal(mean, self.velocity_variance ** 2), 0.1, 1.0)

            notes.append((pitch, velocity, durations[len(notes)]))

        return notes


if __name__ == '__main__':
    composer = Composer(None, None, None, 0.5, 0.001, 0.5, 0.1, 4)
    for i in range(10):
        print(composer.generate_note_sequence())
