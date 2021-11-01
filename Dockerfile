FROM centos
RUN mkdir bd_build
COPY ./ /bd_build/
COPY ./Dependencies/ /bd_build/
RUN bash /bd_build/prepare.sh
RUN bash /bd_build/cleanup.sh
