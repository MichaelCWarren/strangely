FROM rust as builder
WORKDIR /usr/src/strangify
COPY . .
RUN cargo install --path .

FROM debian:bullseye-slim
RUN apt-get update && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/cargo/bin/strangify /usr/local/bin/strangify
CMD ["strangify", "-w"]