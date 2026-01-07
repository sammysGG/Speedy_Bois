# ---------- Builder ----------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    swig \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

RUN pip install --no-cache-dir --prefix=/install \
    "swig>=4.3.1.post0" \
    "gymnasium[box2d,other]==1.2.0" \
    "stable-baselines3[extra]==2.7.0" \
    pyvirtualdisplay \
    jupyter \
    notebook \
    numpy \
    matplotlib


# ---------- Runtime ----------
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only built Python packages
COPY --from=builder /install /usr/local

EXPOSE 8888

# Set Jupyter password
RUN mkdir -p /root/.jupyter && \
    python3 -c "from jupyter_server.auth import passwd; import json; config = {'ServerApp': {'password': passwd('!Number30234872')}}; f = open('/root/.jupyter/jupyter_server_config.json', 'w'); json.dump(config, f); f.close()"

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]