# Docker Files

The software was meant to work on any computer, for  example on a linux platform that didn't have python 3.7 for its python 3 library.
The raspberry pi operating system uses python 3.7.
For these reasons docker containers were employed to implement the scripts from this project.

Containers were designed for both armv7 and amd64 platforms.
The amd64 containers work well.
There are two problems with the armv7 container.
Firstly the neural network models from the project run slower in the container on the raspberry pi than they do if they run natively.
Secondly the container crashed when running tensorflow models.
This was a result of the volume mounting scheme that was used by the container.

* build script - This script moves to the root directory of this repository and uses the Dockerfile there to build a container. It does not work if you don't have docker installed. It uses the `docker build` command.
* launch script - This script launches the docker container as an image. The `docker run` command is very complex. It mounts a filesystem in the docker image so that the project's neural networks can run. Then it launches a bash login shell.
