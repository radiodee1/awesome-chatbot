FROM ubuntu:19.10

RUN apt-get update && apt-get -y install mpg123 python3-pyaudio vim python3 python3-pip curl

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN echo 'deb [arch=amd64] href="http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal"' |  tee /etc/apt/sources.list.d/tensorflow-serving.list
RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg |  apt-key add -
COPY . .
CMD ["ls"]
#EXPOSE 8080
#RUN  /bin/bash
