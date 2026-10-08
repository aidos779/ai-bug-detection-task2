# Immutable multi-architecture index for Python 3.12.10 on Debian bookworm.
FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=42
WORKDIR /app
# No pip install or apt operation: runtime and tests use only the standard library.
COPY src ./src
COPY tests ./tests
COPY configs ./configs
COPY data/sample ./data/sample
RUN python -m unittest discover -s tests -v
CMD ["python", "src/benchmark.py", "--config", "configs/smoke.json"]
