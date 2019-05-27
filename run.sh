#!/usr/bin/env bash

# Config
MIDI_PORT=0

USE_KINECT=false
SYNAPSE=../Synapse-Mac/Synapse.app/Contents/MacOS/Synapse  # Only needed if USE_KINECT is true

# Launch synapse if using kinect
if [[ USE_KINECT == true ]]; then
    $SYNAPSE &
fi

# Setup
source ./setup.sh true

# Run
cd src
python main.py $USE_KINECT $MIDI_PORT
