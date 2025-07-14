#!/bin/bash

export UCX_TLS=cuda_ipc,cuda_copy,tcp
export LMCACHE_LOG_LEVEL=DEBUG

NUM_CHUNKS=500
NUM_ROUNDS=1
PORT=5556

# Bus: ca:00.0
# python3 test_nixl_layerwise_cache_engine.py \
CUDA_VISIBLE_DEVICES=2 python3 test_layer_aware_cache_engine.py \
    --debug \
    --role sender \
    --num-chunks $NUM_CHUNKS \
    --num-rounds $NUM_ROUNDS \
    --port $PORT \
    &
sender_pid=$!  

# Bus: e3:00.0
# python3 test_nixl_layerwise_cache_engine.py \
CUDA_VISIBLE_DEVICES=3 \
python3 test_layer_aware_cache_engine.py \
    --debug \
    --role receiver \
    --num-chunks $NUM_CHUNKS \
    --num-rounds $NUM_ROUNDS \
    --port $PORT \
    &
receiver_pid=$!

wait $sender_pid
wait $receiver_pid

echo "Sender PID: $sender_pid"
echo "Receiver PID: $receiver_pid"
