FROM python:3.12-slim AS base
ARG UV_VERSION=0.5.7
RUN adduser --system --no-create-home nonroot


FROM base AS builder
RUN pip install "uv==${UV_VERSION}"
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
COPY uv.lock pyproject.toml ./
RUN uv pip install -r pyproject.toml

FROM base AS final
WORKDIR /app
COPY --from=builder --chown=app:app /opt/venv /opt/venv
COPY --chown=app:app /streamlit_app.py ./
COPY --chown=app:app /pages ./pages
USER nonroot
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
