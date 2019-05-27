# ShapeSynth

## Summary of Features

* Draw shapes by clicking and holding the canvas.
* Click and hold on a shape to move it around, or drag it offscreen to delete it.
* Each corner of the screen corresponds to a different timbre, resulting in a continuum of tone quality based on the shapes' positions.
* Each shape composes its own music within the global harmony.
* The shape size determines how loud its music will be; the "roughness" of the shape determines its rhythmic complexity.
* You can change the harmony by pressing the 1-6 keys, or by connecting a MIDI keyboard and pressing keys.

## Running ShapeSynth - No Kinect

1. If needed, modify PYTHON_PATH in run.sh. The default is /Library/Frameworks/Python.framework/Versions/3.6/bin/python3. Python 3.5/3.6 is required.
2. If using a MIDI keyboard, you might need to modify MIDI_PORT in run.sh. The default is 0.
2. Make sure you have set `run.sh` to be executable (i.e. `chmod +x run.sh`)
3. Run the script: `./run.sh`.

## Running ShapeSynth - Kinect

Make sure you have Synapse installed on your computer - this is needed to receive Kinect skeleton information.

The `run.sh` shell script launches both Synapse and ShapeSynth in one terminal for convenience (only works on Mac). To use it, follow these instructions:

1. Set USE_KINECT to true in run.sh
2. Make sure the Synapse-Mac directory is in the same directory as the shapesynth directory.
3. Plug the Kinect into your computer.
4. Make sure you have set `run.sh` to be executable (i.e. `chmod +x run.sh`)
5. Run the script: `./run.sh`.
6. Stand at the appropriate distance from the sensor and hold your arms up in the air until Synapse starts tracking your position.
7. To close, quit both ShapeSynth and Synapse, then type `^C` into the Terminal.
