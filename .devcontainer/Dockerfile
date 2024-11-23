FROM python:3.12-slim


ARG REMOTE_USER
ARG REMOTE_UID
ARG REMOTE_GID

# Install git
# RUN  apt-get update \
#   && apt-get install -y git\
#   && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN <<EOF
  addgroup --gid ${REMOTE_GID} ${REMOTE_USER}
  adduser --disabled-password --uid ${REMOTE_UID} --gid ${REMOTE_GID} ${REMOTE_USER}
EOF

ENV HOME /home/${REMOTE_USER}

USER ${REMOTE_USER}

## Install pip dependencies
# Upgrade pip
RUN pip install --upgrade pip
# Install production dependencies.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
