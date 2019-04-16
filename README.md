# shapesynth

## Running Instructions

Make sure you have Synapse installed on your computer - this is needed to receive Kinect skeleton information.

The `run.sh` shell script launches both Synapse and ShapeSynth in one terminal for convenience (only works on Mac). To use it, follow these instructions:

1. Plug the Kinect into your computer.
2. Make sure the `Synapse-Mac` directory is located adjacent to the `shapesynth` directory.
3. Make sure you have set `run.sh` to be executable (i.e. `chmod 777 run.sh`)
4. Run the script: `./run.sh`.
5. Stand at the appropriate distance from the sensor and hold your arms up in the air until Synapse starts tracking your position.
6. To close, quit both ShapeSynth and Synapse, then type `^C` into the Terminal.
