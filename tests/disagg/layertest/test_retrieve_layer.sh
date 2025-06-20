#!/bin/bash
export UCX_TLS=cuda_ipc,cuda_copy,tcp 
export CUDA_VISIBLE_DEVICES=3
export LMCACHE_LOG_LEVEL=DEBUG

python3 test_layer_aware_cache_engine.py --role receiver --num-chunks 256 --num-rounds 5 --port 5556 --debug
