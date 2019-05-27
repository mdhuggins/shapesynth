#!/usr/bin/env bash

# Check that source was used
sourced=true

if [[ $1 != true ]]; then  # Allow override
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
    # Infer python path
    function version_appropriate() { $1 -c "import sys; print(sys.version_info > (3, 5))" 2>/dev/null || echo "false"; }

    # Check common python executable names
    pythonpaths=($(which python) $(which python3) $(which python3.6) $(which py3) $(which py3.6) $(which python3.5) $(which py3.5))

    for path in $pythonpaths; do
        if $(version_appropriate $path); then
            PYTHONPATH=$path
            break
        fi
    done

    if [[ $PYTHONPATH == "" ]]; then
        echo "Couldn't find a compatible Python path. Tried:"
        echo; printf ' * %s\n' "${pythonpaths[@]}"; echo
        echo "ShapeSynth requires Python 3.5 or later."
        while ( ! $(version_appropriate $PYTHONPATH ) ); do
            read -e -p "Enter a path to the Python executable: " PYTHONPATH
        done
    else
        echo "Using Python path $PYTHONPATH"
    fi

    echo "Creating virtual environment..."
    mkdir .virtualenv
    virtualenv --python=$PYTHONPATH --prompt="(ShapeSynth) " .virtualenv
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
