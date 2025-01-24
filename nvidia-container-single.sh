#!/bin/bash

podman run --device nvidia.com/gpu=$1 --shm-size 110G --cap-add=NET_ADMIN --network bridge -it -v ~/DALOS:/workspace/dalos nvcr.io/nvidia/pytorch:23.08-py3