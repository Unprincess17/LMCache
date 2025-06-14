#!/bin/bash
export UCX_TLS=cuda_ipc,cuda_copy,tcp 
export CUDA_VISIBLE_DEVICES=2 

python3 test_layer_aware_cache_engine.py --role sender --num-chunks 8 --num-rounds 1 --port 5556
