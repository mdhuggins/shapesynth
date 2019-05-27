#!/usr/bin/env bash

# Check that source was used
sourced=true

if [[ $2 != true ]]; then  # Allow override
    (return 0 2>/dev/null) && : || sourced=false  # bash
    [[ $ZSH_EVAL_CONTEXT =~ :file$ ]] && : || sourced=false  # zsh

    if $sourced; then
        :
    else
        echo "ERROR: Make sure to call \"source setup.sh\""
        exit 1
    fi
fi

# Check if virtual environment exists
if test -f ".virtualenv/bin/activate"; then
    :
else
    echo "Creating virtual environment..."
    mkdir .virtualenv
    virtualenv --python=$1 --prompt="(ShapeSynth) " .virtualenv
fi

# Load virtual environment
source .virtualenv/bin/activate

# Install dependencies

if [[ $(pip freeze) ]]; then
    :
else
    echo "Installing dependences..."
    pip install -r requirements.txt
fi

echo "Setup done!"