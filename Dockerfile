FROM python:3
USER root

RUN apt-get update && \
    apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y wget bash git gcc g++ gfortran  liblapack-dev libamd2 libcholmod3 libmetis-dev libsuitesparse-dev libnauty2-dev && \
    wget -nH https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew && \
    chmod u+x coinbrew && \
    bash coinbrew fetch Cbc@master && \
    bash coinbrew build Cbc@master --no-prompt --prefix=/usr/local --tests=none --enable-cbc-parallel
ENV PMIP_CBC_LIBRARY="/usr/local/lib/libCbc.so"
ENV LD_LIBRARY_PATH="/home/haroldo/prog/lib"

RUN apt-get install -y vim less && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools

RUN python -m pip install mip && \
    python -m pip install numpy && \
    python -m pip install pandas

# # # # # #

COPY ./opt/__pycache__/main.cpython-310.pyc main.pyc
COPY ./opt/optimize.py optimize.py

CMD python main.cpython-310

# RUN apt-get install  coinor-cbc coinor-libcbc-dev
# RUN python -m pip install jupyterlab
# docker compose up -d --build
# docker exec -it python3cbc bash