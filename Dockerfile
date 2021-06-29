FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime
ENV DEBCONF_NOWARNINGS yes
COPY ./requirements.txt ./

RUN apt-get update && apt-get install -y --quiet --no-install-recommends \
    graphviz \
    wget \
    gcc

# for opencv
RUN apt-get install -y libgl1-mesa-dev

RUN pip install -q --upgrade pip
RUN pip install -r requirements.txt -q

WORKDIR /work
EXPOSE 8888
CMD ["/bin/bash"]
