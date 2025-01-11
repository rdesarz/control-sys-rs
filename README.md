# Control system library 

A control system library written in Rust. This is a work in progress. The goal is to provide a library for control system analysis and synthesis as well as implementation of control algorithms.

## Steps

1. Develop functions to build state-space representation of a control system.
2. Develop functions to analyse those systems (observability, controllability, poles, etc)

## Development

Development can be made in a Docker container following these steps: 

```
docker compose build
docker compose up -d
```

Then run the code inside the container:

```
cargo run
docker exec -it control-sys-rs bash
```
