#!/bin/bash

# Define configurations
configs=(
  "--config-path pkg://sn_gamestate.configs --config-name soccernet_test"
  "--config-path pkg://sn_gamestate.configs --config-name soccernet_test1"
  "--config-path pkg://sn_gamestate.configs --config-name soccernet_test2"
  "--config-path pkg://sn_gamestate.configs --config-name soccernet_test3"
)

# Iterate through configurations and run in parallel
for config in "${configs[@]}"; do
    echo "Running with config: ${config}"
    python /content/sn-gamestate/main.py ${config} &
done

# Wait for all background processes to complete
wait

echo "All processes finished."
