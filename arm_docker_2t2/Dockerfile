
#ARG LS

FROM emacski/tensorflow-serving:latest-arm32v7

ARG LS=*
ARG EXPORT=saved/t2t_train/chat_10/export/*

#COPY ${EXPORT} ${EXPORT}
COPY . .
CMD ["echo","hello docker!"]
ENTRYPOINT ["echo", "${LS}"]


RUN ls ${LS}
RUN echo -----
RUN ls

#RUN echo -----
#FROM emacski/tensorflow-serving:latest-arm32v7 #tensorflow/serving:$TFS_VERSION-devel as devel
# install arm toolchain dependencies

#COPY tensorflow_serving/tools/docker/install_arm_toolchain.sh /install_arm_toolchain.sh
#RUN /install_arm_toolchain.sh
# overwrite existing sources
# adds config groupings with default options for arm32v7 and arm64v8

# Install TF Serving binary
#COPY --from=build /tensorflow-serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server /usr/bin/tensorflow_model_server
# Expose ports
# gRPC

# Set where models should be stored in the container
#ENV MODEL_BASE_PATH=/models
#RUN mkdir -p ${MODEL_BASE_PATH}
# The only required piece is the model name in order to differentiate endpoints
#ENV MODEL_NAME=model
#ENTRYPOINT tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}