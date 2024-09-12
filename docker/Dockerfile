FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y r-base

ADD https://api.github.com/repos/shahashka/causal_discovery_via_partitioning/git/refs/heads/main version.json
RUN git clone https://github.com/shahashka/causal_discovery_via_partitioning.git

WORKDIR /causal_discovery_via_partitioning

RUN conda update -n base conda \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda env create --file env.yml

SHELL ["conda", "run", "--no-capture-output", "-n", "causal_discovery", "/bin/bash", "-c"]

RUN pip install -e . -r docker/requirements.txt
RUN Rscript docker/requirements.R
RUN conda install -c conda-forge gcc=12.1.0

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "causal_discovery"]
