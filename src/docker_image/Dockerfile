# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ARG MINICONDA_VERSION=latest
FROM continuumio/miniconda3:$MINICONDA_VERSION

RUN apt-get update -y; \
    apt-get upgrade -y; \
    apt-get install -y 

RUN apt-get install wget -y

RUN  conda update conda \
    && conda clean --all --yes

USER $USER

WORKDIR /maindir

# RUN mkdir /maindir/data

# COPY goes_env.yml .
COPY goes/goes_env.yml .

# Create environment
RUN conda env create -f goes_env.yml

# Add extra dependencies for opencv (cv2)
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# Make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# Make non-activate conda commands available
ENV PATH=$CONDA_DIR/envs/goes_env/bin:$PATH

# RUN source ~/.profile
# Start bash with conda env
RUN echo "conda activate goes_env" >> ~/.bashrc
ENV PATH /opt/conda/envs/goes_env/bin/python:$PATH
ENV CONDA_DEFAULT_ENV $goes_env

# Install aws cli
RUN apt install curl -y
RUN apt install unzip -y
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf /maindir/awscliv2.zip

# Copy Python scripts
# COPY A_download_goes.py .
# COPY B_process_goes.py .
# COPY C_predict_goes.py .
# COPY D_clean_goes.py .


COPY goes/src ./src
COPY goes/config ./config
COPY utils ./utils


# Create container entry point (Where it will start every time we run teh container)
# ENTRYPOINT /bin/bash --login -c "conda activate test_env && python3 example_time.py"

# Another option is to use conda run but its use is discourage as it is still in development.
# ENTRYPOINT [ "conda", "run", "--no-capture-output", "-n", "test_env", "python3", "example_time.py"]