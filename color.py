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
        self.hue_average = 0.5
        self.hue_range = (0, 1)
        self.brightness_range = (0.5, 1)

    def process_harmony(self, harmony):
        """
        Updates the hue range when the harmony changes.
        """
        # Sample hue using a normal distribution centered around major/minor balance
        hue_average = np.mod(np.random.uniform(self.hue_average - 0.2, self.hue_average + 0.2), 1.0)

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
