FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y r-base

RUN git clone https://github.com/shahashka/causal_discovery_via_partitioning.git

WORKDIR /causal_discovery_via_partitioning

RUN conda update -n base conda \
    && conda install -n base conda-libmamba-solver \
    && conda config --set solver libmamba \
    && conda env create --file ./docker/env.yml

SHELL ["conda", "run", "--no-capture-output", "-n", "causal_discovery", "/bin/bash", "-c"]

RUN pip install -e . -r docker/requirements.txt
RUN Rscript docker/requirements.R

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "causal_discovery", "python", "/causal_discovery_via_partitioning/simulations/experiment_basecase.py"]
