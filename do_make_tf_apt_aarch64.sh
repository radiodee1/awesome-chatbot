sudo apt-get update \
    && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y mpg123 python3-pyaudio vim python3 python3-pip curl mpg321 \
    alsa-utils  libasound2-plugins wget build-essential python-opencv python3-opencv cython3 cython python3-scipy \
    python3-matplotlib python3-cffi python3-greenlet python3-pycparser python3-gevent  python3-h5py \
    libxml2-dev libxslt-dev python3-lxml libopenblas-dev pciutils alsa-base libhdf5-dev libhd-dev apt-transport-https \
    ca-certificates portaudio19-dev libssl-dev python3-pygame 


sudo apt-get -y remove python3-mpi4py

#sudo pip3 install tensor2tensor==1.15.5 tensorflow-serving-api==1.15.0