docker run --privileged -d arm32v7/docker:dind --name arm_dind

docker run -it --rm --privileged --link arm32v7/docker:dind -d arm_dind ls
