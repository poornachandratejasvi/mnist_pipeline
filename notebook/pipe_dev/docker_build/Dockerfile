FROM centos
RUN mkdir bd_build
COPY ./../py_pipe_dev /bd_build/
COPY ./prepare.sh /bd_build/
COPY ./cleanup.sh /bd_build/
COPY ./requirements.txt /bd_build/
COPY ./deployment/dockerfile/csh/Dependencies/ /bd_build/
RUN bash /bd_build/prepare.sh
RUN bash /bd_build/cleanup.sh
