# Docker file for hosting hvac_gym. Builds your project and its dependencies into a docker container.
# This is mostly just the filesystem setup. Use docker-compose for runtime stuff (eg volumes, actual entrypoints etc)

ARG PROJECT_NAME=hvac_gym
ARG PYTHON_VERSION=3.12.1
ARG POETRY_VERSION=1.8.3

FROM python:$PYTHON_VERSION-slim

# Arguments defined before a FROM will be unavailable after the FROM unless we re-specify the arg
ARG POETRY_VERSION
ARG PROJECT_NAME
ARG PUBLISH_URL
ARG PUBLISH_USER
ARG PUBLISH_PASSWORD

WORKDIR /usr/src/app/$PROJECT_NAME/

# install poetry
RUN apt-get update && apt-get install curl git -y

# Extra Tools For debugging
#RUN apt-get install nano iputils-ping net-tools nmap -y

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="/root/.local/bin/:${PATH}"

# Add private repo if relevant args exist
RUN bash -c 'if [[ -z "$PUBLISH_URL" ]] ;  then echo ${PUBLISH_URL} not provided ;  else poetry config repositories.csiroenergy $PUBLISH_URL ; fi'
RUN bash -c 'if [[ -z "$PUBLISH_USER" ]] ; then echo ${PUBLISH_USER} not provided ; else poetry config http-basic.csiroenergy $PUBLISH_USER $PUBLISH_PASSWORD ; fi'
RUN poetry config --list

# set up python environment
COPY poetry.lock ./
RUN poetry config virtualenvs.in-project true
RUN poetry config installer.max-workers 10 # avoid 'Connection pool is full' errors

# Copy dependency list and install
COPY pyproject.toml ./
RUN poetry install --no-root --sync

# Now copy our own source and "install" it (adds to pythonpath) - we do this as a separate step to avoid reinstalling all dependencies if only our source changes
# Note: make sure anything to skip is added to .dockerignore
COPY ./ ./
RUN poetry install

# just launch bash for debugging. Specify real entrypoints in docker-compose.yml
ENTRYPOINT ["/bin/bash"]
