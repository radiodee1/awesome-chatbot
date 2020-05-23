cd / && git clone https://github.com/tensorflow/tensorflow.git

cd /tensorflow && git checkout r1.15 && ls

cd /tensorflow && CI_DOCKER_EXTRA_PARAMS="--privileged -e CI_BUILD_PYTHON=python3.7 -e CROSSTOOL_PYTHON_INCLUDE_PATH=/usr/include/python3.7" \
    tensorflow/tools/ci_build/ci_build.sh PI-PYTHON3 \
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh

cd /tensorflow && tensorflow/tools/ci_build/ci_build.sh PI \
    tensorflow/tools/ci_build/pi/build_raspberry_pi.sh PI_ONE

ls

bash --login