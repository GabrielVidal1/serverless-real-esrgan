# This is a potassium-standard dockerfile, compatible with Banana

# Don't change this. Currently we only support this specific base image.
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install git wget ffmpeg libsm6 libxext6 -y

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN git clone https://github.com/xinntao/Real-ESRGAN.git realesrgan
RUN echo "from .realesrgan import *" > realesrgan/__init__.py
RUN pip3 install -r realesrgan/requirements.txt

# Add your model weight files 
# (in this case we have a python script)
RUN wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P realesrgan/weights

ADD . .

EXPOSE 8000

CMD python3 -u app.py