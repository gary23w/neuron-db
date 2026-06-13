# Build the neuron-db server + CLI, ship a slim runtime image.
FROM rust:1-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY rust/neuron-core ./neuron-core
WORKDIR /src/neuron-core
RUN cargo build --release --features "sqlite secure server" --bin serve --bin neuron

FROM debian:bookworm-slim
RUN useradd -m neuron && mkdir -p /data && chown neuron /data
COPY --from=build /src/neuron-core/target/release/serve  /usr/local/bin/serve
COPY --from=build /src/neuron-core/target/release/neuron /usr/local/bin/neuron
USER neuron
# DB lives on a volume; bind 0.0.0.0 so the port is reachable from outside the container.
# Set NEURON_DB_KEY to require Authorization: Bearer <key>.
ENV NEURON_DB=/data/neurons.db NEURON_HOST=0.0.0.0 NEURON_PORT=8088
VOLUME /data
EXPOSE 8088
ENTRYPOINT ["serve"]
