#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: ./create_new_experiment.sh <experiment_name>"
    echo ""
    echo "Example: ./create_new_experiment.sh my_new_experiment"
    read -p "Press Enter to continue..."
    exit 1
fi

python ../python_scripts/create_new_experiment.py "$1"
