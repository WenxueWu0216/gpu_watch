# Bring-up gate image: nvidia-smi comes from the host via the NVIDIA container
# runtime; this image adds DCGM (dcgmi), nccl-tests and kubectl.
# NOT VERIFIED IN CI (needs GPU hosts to be meaningful). Build:
#   docker build -f deploy/docker/bringup.Dockerfile -t gpuwatch-bringup .
FROM nvcr.io/nvidia/cuda:12.4.1-devel-ubuntu22.04
ARG NCCL_TESTS_REF=v2.13.11
ARG KUBECTL_VERSION=v1.31.2
RUN apt-get update && apt-get install -y --no-install-recommends \
      datacenter-gpu-manager git make python3 python3-pip python3-venv curl ca-certificates \
      libnccl2 libnccl-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${NCCL_TESTS_REF} https://github.com/NVIDIA/nccl-tests /opt/nccl-tests \
    && make -C /opt/nccl-tests -j"$(nproc)" \
    && ln -s /opt/nccl-tests/build/all_reduce_perf /usr/local/bin/all_reduce_perf
RUN curl -fsSLo /usr/local/bin/kubectl https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl \
    && chmod +x /usr/local/bin/kubectl
WORKDIR /opt/gpuwatch
COPY pyproject.toml README.md ./
COPY gpu_sim gpu_sim
COPY detector detector
COPY bringup bringup
COPY tracing tracing
COPY ai_diagnosis ai_diagnosis
COPY eval eval
RUN pip3 install --no-cache-dir ".[otel]"
# nv-hostengine must run for dcgmi diag; start it embedded per invocation.
ENTRYPOINT ["bash", "-c", "nv-hostengine >/dev/null 2>&1 || true; exec python3 -m bringup \"$@\"", "--"]
