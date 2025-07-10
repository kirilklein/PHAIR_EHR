#!/bin/bash

run_step() {
  module="$1"; config="$2"
  echo "🔧 Running corebehrt.$module ($config)"
  python -m corebehrt.$module \
    --config_path corebehrt/configs/causal/$config
}

run_test() {
  margin="$1"; dir="$2"
  echo "✅ Checking estimation in $dir"
  python -m tests.test_main_causal.test_estimate_result \
    --margin "$margin" \
    --dir "./outputs/causal/$dir"
}