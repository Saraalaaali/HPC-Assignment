FROM python:3.9-slim

# 1. Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin \
    libopenmpi-dev \
    openssh-server \
    openssh-client \
    gcc \
    g++ \
    make \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mpi4py opencv-python-headless numpy matplotlib

# 2. SSH Setup (Organized like your reference image)
# Create directories & Set Root password
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh && \
    echo 'root:root' | chpasswd

# Generate System Host Keys (Essential)
RUN /usr/bin/ssh-keygen -A

# Generate User Keys for passwordless login
RUN ssh-keygen -t rsa -b 2048 -f /root/.ssh/id_rsa -N "" -q && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys /root/.ssh/id_rsa

# Configure SSH Client (No prompts)
RUN printf "Host *\n    StrictHostKeyChecking no\n    UserKnownHostsFile /dev/null\n    LogLevel ERROR\n" \
    > /root/.ssh/config && \
    chmod 600 /root/.ssh/config

# Configure SSH Daemon (Server settings)
RUN printf "Port 22\nPermitRootLogin yes\nPubkeyAuthentication yes\n\
AuthorizedKeysFile /root/.ssh/authorized_keys\nPasswordAuthentication yes\n\
PermitEmptyPasswords no\nChallengeResponseAuthentication no\nUsePAM no\n" \
    > /etc/ssh/sshd_config

# 3. App Setup
WORKDIR /app
COPY . .

# 4. Compile C++ Code
RUN set -e; \
    if pkg-config --exists opencv4 2>/dev/null; then \
        FLAGS=$(pkg-config --cflags --libs opencv4); \
    elif pkg-config --exists opencv 2>/dev/null; then \
        FLAGS=$(pkg-config --cflags --libs opencv); \
    else \
        FLAGS="-I/usr/include/opencv4 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs"; \
    fi; \
    mpicxx task2_histogram_mpi.c -o task2_histogram_mpi $FLAGS -lm && \
    echo "Compiled OK"

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]