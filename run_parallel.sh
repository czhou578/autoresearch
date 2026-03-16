#!/bin/bash

# Run worker 1
(   echo "=== WORKER 1 ===" 
    cd ../worker1
    python -u train.py
) > /workspace/autoresearch/worker1.log 2>&1 &
PID1=$!

# Run worker 2
(   echo "=== WORKER 2 ==="
    cd ../worker2
    python -u train.py
) > /workspace/autoresearch/worker2.log 2>&1 &
PID2=$!

wait $PID1
wait $PID2

cat /workspace/autoresearch/worker1.log >> /workspace/autoresearch/run.log
cat /workspace/autoresearch/worker2.log >> /workspace/autoresearch/run.log
