#!/bin/bash

export UCX_TLS=cuda_ipc,cuda_copy,tcp

# Bus: ca:00.0
CUDA_VISIBLE_DEVICES=2 \
python3 ../test_nixl_cache_engine.py \
    --role sender \
    --num-chunks 500 \
    --num-rounds 1 \
    --port 5556 \
    &
sender_pid=$!  

# Bus: e3:00.0
CUDA_VISIBLE_DEVICES=3 \
python3 ../test_nixl_cache_engine.py \
    --role receiver \
    --num-chunks 500 \
    --num-rounds 1 \
    --port 5556

wait $sender_pid
