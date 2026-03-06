#!/bin/bash

show_help() {
    echo "Usage: $0 <path_to_model.pt> <grid_size>"
    echo "Example: $0 /path/to/model.pt 30"
    exit 1
}

if [ $# -lt 2 ] || [[ " $@ " =~ " --help " ]]; then
    show_help
fi

LULESH_DIR=/mnt/SharedOne/bpanthi/LULESH
SCRIPT_DIR=/mnt/SharedOne/bpanthi/model_search

model=$1 # Path to the model.pt file
model_name=$(basename "$(dirname "$model")")
grid_size=$2

experiment_dir="${model_name}-${grid_size}"
mkdir -p "$experiment_dir"

start_time=$(date +%s.%N)
SURROGATE_MODEL=$model ENERGY_DUMP_FILE_NAME=$experiment_dir/Energy-$model_name-$grid_size.bin ENERGY_DUMP_TYPE=last ./lulesh2.0 -p -s $grid_size
end_time=$(date +%s.%N)

# Store time taken in execution_time.txt file

time_diff=$(echo "$end_time - $start_time" | bc)
total_seconds=${time_diff%.*}
[ -z "$total_seconds" ] && total_seconds=0
h=$((total_seconds / 3600))
m=$(( (total_seconds % 3600) / 60 ))
s=$(( total_seconds % 60 ))
formatted_time=$(printf "%02d:%02d:%02d" $h $m $s)
echo "$time_diff seconds" > "$experiment_dir/execution_time.txt"
echo "$formatted_time" >> "$experiment_dir/execution_time.txt"


source $SCRIPT_DIR/.venv/bin/activate

# Print the output of the following program and store the output in $experiment_dir/energy_mae.txt
python $SCRIPT_DIR/energy.py --original $LULESH_DIR/Energy_Original-$grid_size.bin --model "$experiment_dir/Energy-$model_name-$grid_size.bin" --visualize | tee "$experiment_dir/energy_mae.txt"
