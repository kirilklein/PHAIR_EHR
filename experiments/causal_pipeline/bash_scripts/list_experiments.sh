#!/bin/bash

echo "========================================"
echo "Available Causal Pipeline Experiments"
echo "========================================"
echo ""

if ! ls ../experiment_configs/*.yaml 1> /dev/null 2>&1; then
    echo "No experiments found in ../experiment_configs/"
    echo ""
    echo "Create a new experiment with: ./create_new_experiment.sh <name>"
    read -p "Press Enter to continue..."
    exit 0
fi

echo "Experiment Name         Description"
echo "----------------       -----------"

for file in ../experiment_configs/*.yaml; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .yaml)
        
        # Try to extract description from YAML file
        description=$(grep "description:" "$file" 2>/dev/null | sed 's/description:[[:space:]]*//' | tr -d '"' | head -1)
        
        if [ -z "$description" ]; then
            description="No description"
        fi
        
        printf "%-23s %s\n" "$filename" "$description"
    fi
done

echo ""
echo "Usage Examples:"
echo "  ./run_experiment.sh <experiment_name>"
echo "  ./run_experiment_full.sh <experiment_name>"
echo "  ./run_all_experiments.sh"
echo "  ./run_all_experiments_full.sh"
echo "  ./run_experiments_ordered.sh exp1 exp2 exp3"
echo "  ./create_new_experiment.sh <new_experiment_name>"
echo ""
read -p "Press Enter to continue..."
