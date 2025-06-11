#!/bin/bash
export UCX_TLS=cuda_ipc,cuda_copy,tcp 
export CUDA_VISIBLE_DEVICES=3

python3 test_nixl_layerwise_cache_engine.py --role receiver --num-chunks 500 --num-rounds 1 --port 5556
