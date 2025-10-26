#!/bin/bash

# ========================================
# Monitor Experiment Progress
# ========================================

echo "========================================"
echo "Experiment Progress Monitor"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "Experiment Progress Monitor"
    echo "========================================"
    echo "Current Time: $(date)"
    echo ""
    
    # Check for running Python processes
    PYTHON_RUNNING=$(pgrep -f python | wc -l)
    
    if [ $PYTHON_RUNNING -gt 0 ]; then
        echo "Status: EXPERIMENTS RUNNING ($PYTHON_RUNNING Python processes)"
    else
        echo "Status: NO EXPERIMENTS RUNNING"
    fi
    echo ""
    
    # Show recent experiment outputs
    echo "Recent Experiment Directories:"
    if [ -d "../../outputs/causal/sim_study/runs/" ]; then
        ls -1t "../../outputs/causal/sim_study/runs/" 2>/dev/null | head -10 | while read dir; do
            echo "  $dir"
        done
    else
        echo "  No experiment outputs found yet"
    fi
    echo ""
    
    # Show recent log files
echo "Recent Log Files:"
if ls ../logs/*.log 1> /dev/null 2>&1; then
    ls -1t ../logs/*.log 2>/dev/null | head -5 | while read file; do
        echo "  $(basename "$file")"
    done
else
    echo "  No log files found in ../logs/"
fi
    echo ""
    
    echo "Press Ctrl+C to exit, or wait for auto-refresh..."
    sleep 10
done
