FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN pip install taker==1.0.2 hqq bitsandbytes

