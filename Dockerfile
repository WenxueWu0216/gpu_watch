# Python image for gpu_sim, detector, bringup and ai_diagnosis (sim stack).
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY gpu_sim gpu_sim
COPY detector detector
COPY bringup bringup
COPY tracing tracing
COPY ai_diagnosis ai_diagnosis
COPY eval eval
RUN pip install --no-cache-dir ".[otel]"
RUN useradd -u 10001 gpuwatch
USER gpuwatch
ENTRYPOINT ["python", "-m"]
CMD ["gpu_sim", "serve"]
