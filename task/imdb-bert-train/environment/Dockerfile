ARG BASE_IMAGE=local/flagscale-agent-base-rewardkit:cu128-py312
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN python3 --version && \
    python3 -m pip --version && \
    # python3 -m pip install --no-cache-dir uv && \
    # uv tool install harbor-rewardkit && \
    # rewardkit --help >/dev/null
    python3 -m venv /tmp/venv-check && \
    /tmp/venv-check/bin/python --version && \
    rm -rf /tmp/venv-check

CMD ["bash", "-lc", "sleep infinity"]

