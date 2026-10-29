# SPDX-FileCopyrightText: 2026-present Mario Abarca <knkillname@gmail.com>
#
# SPDX-License-Identifier: MIT
#
# Track the latest Colab CPU runtime so the dev container stays current. The
# separate Colab smoke gate remains pinned by digest for reproducibility.
FROM us-docker.pkg.dev/colab-images/public/cpu-runtime:latest

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_LOG=/tmp/pip.log \
    PYTHONPATH= \
    UV_BREAK_SYSTEM_PACKAGES=false \
    UV_SYSTEM_PYTHON=false \
    UV_TOOL_DIR=/usr/local/share/uv/tools \
    UV_TOOL_BIN_DIR=/usr/local/bin

# Colab starts Jupyter through /datalab/run.sh. A dev container needs a
# long-lived shell instead; Docker-in-Docker adds its own entrypoint later.
ENTRYPOINT []
CMD ["sleep", "infinity"]

# Hatch drives the repository checks; OpenCode is the in-container agent.
RUN uv tool install --python /usr/local/bin/python hatch \
    && curl -fsSL https://opencode.ai/v2/install | bash -s -- --no-modify-path \
    && install -m 0755 /root/.opencode/bin/opencode /usr/local/bin/opencode \
    && rm -rf /root/.opencode

# Colab ships the non-root 'ubuntu' user at UID 1000. Dev Containers uses it
# as the remote user; passwordless sudo is needed by editor tooling.
RUN printf 'ubuntu ALL=(ALL) NOPASSWD:ALL\n' > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu
