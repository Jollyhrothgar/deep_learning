FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN  python -m pip install -r requirements.txt
