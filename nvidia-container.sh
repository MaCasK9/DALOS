#!/bin/bash

podman run --device nvidia.com/gpu=all --shm-size 64G -it -v ~/data/DALOS:/workspace/dalos nvcr.io/nvidia/pytorch:23.08-py3 
