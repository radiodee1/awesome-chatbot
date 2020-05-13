FROM ubuntu:19.10
#WORKDIR /code
#ENV FLASK_APP app.py
#ENV FLASK_RUN_HOST 0.0.0.0
RUN apt-get update && apt-get -y install mpg123 python3-pyaudio vim python3 python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
#CMD ["flask", "run"]
RUN -it -p 8080:80  ubuntu /bin/bash
