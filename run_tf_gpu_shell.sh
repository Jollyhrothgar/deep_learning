#!/bin/bash

docker run -it -u $(id -u):$(id -g) \
	--gpus all -p 8888:8888 \
	-v "${HOME}/.config/jupyter:/.jupyter" \
	-v "${HOME}/workspace:/tf" \
	tensorflow/tensorflow:latest-gpu-jupyter \
	bash 
