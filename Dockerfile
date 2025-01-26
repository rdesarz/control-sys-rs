FROM rust:latest

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --allow-unauthenticated --no-install-recommends \
        liblapack-dev \
        libopenblas-dev \
        gfortran \
        build-essential \
        sudo 

# Install rustfmt
RUN rustup component add rustfmt

# Install cargo generate
RUN cargo install cargo-generate

# Copy the project inside the image
COPY . /root/control_sys

WORKDIR /root/control_sys
