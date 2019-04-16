#pset5: Magic Harp

# common import
import sys
sys.path.append('..')

from common.core import *
from common.gfxutil import *
from common.audio import *
from common.synth import *
from common.clock import *

from kivy.graphics import Color, Line
from kivy.graphics.instructions import InstructionGroup
from kivy.clock import Clock as kivyClock

import numpy as np

from common.kinect import *

# x, y, and z ranges to define a 3D bounding box
kKinectRange = ( (-500, 500), (-200, 700), (-500, 0) )

class MainWidget(BaseWidget) :
    def __init__(self):
        super(MainWidget, self).__init__()

        self.info = topleft_label()
        self.add_widget(self.info)

        self.audio = Audio(2)
        #self.synth = Synth("../data/FluidR3_GM.sf2")
        #self.audio.set_generator(self.synth)

        # Set up kinect
        self.kinect = Kinect()
        self.kinect.add_joint(Kinect.kLeftHand)
        self.kinect.add_joint(Kinect.kRightHand)

        # Create cursors
        self.margin = np.array([Window.width * 0.05, Window.width * 0.05])
        self.window_size = [Window.width - 2 * self.margin[0], Window.height - 2 * self.margin[1]]
        self.left_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5))
        self.canvas.add(self.left_hand)
        self.right_hand = Cursor3D(self.window_size, self.margin.tolist(), (0.5, 0.5, 0.5))
        self.canvas.add(self.right_hand)

    def on_update(self) :
        self.info.text = ''

        self.kinect.on_update()
        pt1 = self.kinect.get_joint(Kinect.kRightHand)
        pt2 = self.kinect.get_joint(Kinect.kLeftHand)
        norm_left = scale_point(pt2, kKinectRange)
        norm_right = scale_point(pt1, kKinectRange)

        self.left_hand.set_pos(norm_left)
        self.right_hand.set_pos(norm_right)
        self.audio.on_update()

        #self.info.text += 'x:%d\ny:%d\nz:%d\n' % tuple(pt.tolist())


# pass in which MainWidget to run as a command-line arg
run(MainWidget, title="ShapeSynth")
