# shapesynth

## Summary of Features

* Draw shapes by clicking and holding the canvas.
* Click and hold on a shape to move it around, or drag it offscreen to delete it.
* Each corner of the screen corresponds to a different timbre, resulting in a continuum of tone quality based on the shapes' positions.
* Each shape composes its own music within the global harmony.
* The shape size determines how loud its music will be; the "roughness" of the shape determines its rhythmic complexity.
* You can change the harmony by pressing the 1-6 keys, or by connecting a MIDI keyboard and pressing keys. *Note:* If the keyboard is not recognized, you may need to change the MIDI port. For example, to change to port 1, start the application with `python main.py False 1`.

## Install

Run the following command to install any required dependencies: `pip install -r requirements.txt`

## Running Instructions - No Kinect

```
python main.py
```

## Running Instructions - Kinect

Make sure you have Synapse installed on your computer - this is needed to receive Kinect skeleton information.

The `run.sh` shell script launches both Synapse and ShapeSynth in one terminal for convenience (only works on Mac). To use it, follow these instructions:

1. Plug the Kinect into your computer.
2. Make sure the `Synapse-Mac` directory is located adjacent to the `shapesynth` directory.
3. Make sure you have set `run.sh` to be executable (i.e. `chmod 777 run.sh`)
4. Run the script: `./run.sh`.
5. Stand at the appropriate distance from the sensor and hold your arms up in the air until Synapse starts tracking your position.
6. To close, quit both ShapeSynth and Synapse, then type `^C` into the Terminal.
