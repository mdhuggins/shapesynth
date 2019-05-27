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

# Infer python path
if [ "$(cat .shapesynth_settings 2>/dev/null || echo "")" != "" ]; then
  PYTHONPATH=$(cat .shapesynth_settings)
else
  function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }
  function version_appropriate() { ! version_gt "3.5.0" $1; }
  function get_python_version() { $1 -c "import platform; print(platform.python_version())" 2>/dev/null || echo "1.0"; }

  # Check common python executable names
  pythonversions=($(which python) $(which python3) $(which python3.6) $(which py3) $(which py3.6) $(which python3.5) $(which py3.5))

  for version in $pythonversions; do
    if $(version_appropriate $(get_python_version $version)); then
      PYTHONPATH=$version
      break
    fi
  done

  if [[ $PYTHONPATH == "" ]]; then
    echo "Couldn't find a compatible Python path. Tried:"
    echo; printf ' * %s\n' "${pythonversions[@]}"; echo
    while ( ! $(version_appropriate $(get_python_version $PYTHONPATH)) ); do
      read -e -p "ShapeSynth requires Python 3.5 or later. Enter a path to the Python executable: " PYTHONPATH
      if $(version_appropriate $(get_python_version $PYTHONPATH)); then
        echo "Appropriate!"
      fi
    done
  else
    echo "Using Python path $PYTHONPATH"
  fi

  echo $PYTHONPATH > .shapesynth_settings
fi

# Check if virtual environment exists
if test -f ".virtualenv/bin/activate"; then
    :
else
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
