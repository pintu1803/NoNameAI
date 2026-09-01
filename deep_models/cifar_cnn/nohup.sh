#!/bin/bash

# Create log directory if not exists
mkdir -p logs

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

LOG_FILE="logs/train_${TIMESTAMP}.log"

echo "Starting training..."
echo "Logs: ${LOG_FILE}"

# Run in background
echo "nohup python -u main.py > ${LOG_FILE} 2>&1 &"
nohup python -u main.py > ${LOG_FILE} 2>&1 &

# Save PID
echo $! > train.pid

echo "Training started with PID $(cat train.pid)"

echo "You can tail the log file now"
echo "tail -f ${LOG_FILE}"
