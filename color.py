from composer import Conductor
import numpy as np

class ColorPalette(object):
    """
    A type that reads the current conductor's harmony to determine an appropriate
    color for each shape on the canvas.
    """
    def __init__(self):
        super(ColorPalette, self).__init__()
        self.cache_harmony = None
        self.hue_variance = 0.1
        self.hue_range = (0, 1)
        self.brightness_range = (0.5, 1)
        self.major_mat = self.roll_vector(np.array([4, 0, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1]))
        self.minor_mat = self.roll_vector(np.array([4, 0, 1, 2, 0, 1, 0, 2, 1, 0, 1, 0]))

    def roll_vector(self, vec):
        """Creates a matrix of all possible horizontal shifts of the given vector."""
        mat = np.tile(vec, (12, 1))
        new_mat = np.zeros_like(mat)
        for n in range(mat.shape[0]):
            new_mat[n] = np.roll(mat[n], n)
        return new_mat

    def process_harmony(self, harmony):
        """
        Updates the hue range according to the given harmony (set of pitch classes).
        """
        harmony_vector = np.zeros(12)
        for pitch in harmony:
            if pitch < 0 or pitch >= 12: continue
            harmony_vector[pitch] = 1
        harmony_vector[min(harmony)] = 2
        major_coef = np.max(np.dot(self.major_mat, harmony_vector))
        minor_coef = np.max(np.dot(self.minor_mat, harmony_vector))

        # Sample hue using a normal distribution centered around major/minor balance
        hue_average = ((minor_coef / (major_coef + minor_coef)) - 0.3) / (0.7 - 0.3)
        hue_average = np.random.normal(hue_average, 0.4)

        self.hue_range = (np.clip(hue_average - self.hue_variance, 0.0, 1.0), np.clip(hue_average + self.hue_variance, 0.0, 1.0))
        self.cache_harmony = harmony

    def new_color(self, saturation_parameter):
        """
        Returns a new random color as an (h, s, v) tuple, generated based on the
        conductor's current harmony. saturation_parameter should be in [0, 1] and
        specifies the approximate range of saturation to use for the color.
        """
        if Conductor.harmony != self.cache_harmony:
            self.process_harmony(Conductor.harmony)
        return (np.random.uniform(*self.hue_range), np.clip(np.random.normal(saturation_parameter * 2.0, 0.2), 0.0, 1.0), np.random.uniform(*self.brightness_range))
