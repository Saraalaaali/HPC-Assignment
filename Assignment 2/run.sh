#!/usr/bin/env bash
# run.sh — MPI Cluster management for Task 2 (Histogram)

set -e
cd "$(dirname "$0")"

# ── Create required directories ──────────────────────────────────────────────
mkdir -p images output

# ── Image selector ───────────────────────────────────────────────────────────
if [ -z "$MPI_IMAGE" ]; then
  echo ""
  echo "╔══════════════════════════════════════════╗"
  echo "║      MPI Cluster — Image Manager         ║"
  echo "╚══════════════════════════════════════════╝"
  echo ""
  mapfile -t IMAGES < <(docker images --format "{{.Repository}}:{{.Tag}}" \
    | grep -v "<none>" | sort -u)

  for i in "${!IMAGES[@]}"; do
    echo "  [$((i+1))] ${IMAGES[$i]}"
  done
  echo "  [b] Build new image from Dockerfile"
  echo ""
  read -rp "Choose [number/b]: " CHOICE

  if [[ "$CHOICE" == "b" ]]; then
    MPI_IMAGE="mpi_app:latest"
    docker compose build --no-cache
  elif [[ "$CHOICE" =~ ^[0-9]+$ ]] && [ "$CHOICE" -ge 1 ] && [ "$CHOICE" -le "${#IMAGES[@]}" ]; then
    MPI_IMAGE="${IMAGES[$((CHOICE-1))]}"
  else
    echo "Invalid choice — defaulting to mpi_app:latest"
    MPI_IMAGE="mpi_app:latest"
  fi
fi

export MPI_IMAGE

# ── Start the cluster ─────────────────────────────────────────────────────────
echo ""
echo "Starting cluster with image: $MPI_IMAGE ..."
docker compose down --remove-orphans
# ── Start the cluster ────────────────────────────────────────────────────────
echo "Starting MPI cluster..."
docker compose up -d

echo "Waiting for SSH to be ready (5 seconds)..."
sleep 5

# ── Force-inject the hostfile (Solves the recognition issue) ──────────────────
echo "Injecting hostfile into Master..."
docker exec master bash -c "cat > /app/hostfile << EOF
localhost slots=1
worker1 slots=1
worker2 slots=1
worker3 slots=1
worker4 slots=1
worker5 slots=1
worker6 slots=1
EOF"

# ── Verify connectivity ───────────────────────────────────────────────────────
echo "Checking SSH connectivity..."
for HOST in worker1 worker2 worker3 worker4 worker5 worker6; do
  if docker exec master ssh -o ConnectTimeout=3 "$HOST" "echo OK" &>/dev/null; then
    echo "  ✓ $HOST reachable"
  else
    echo "  ✗ $HOST NOT reachable - restarting SSH on $HOST"
    docker exec "$HOST" service ssh restart
  fi
done
echo "Waiting for SSH to be ready (5 seconds)..."
sleep 5

# ── Verify all workers are reachable ─────────────────────────────────────────
echo ""
echo "Checking SSH connectivity to workers..."
for HOST in worker1 worker2 worker3 worker4 worker5 worker6; do
  if docker exec master ssh -o ConnectTimeout=3 "$HOST" "echo OK" &>/dev/null; then
    echo "  ✓ $HOST reachable"
  else
    echo "  ✗ $HOST NOT reachable — check network config"
  fi
done

# ── Usage instructions ────────────────────────────────────────────────────────
echo ""
echo "✓ Cluster ready (1 master + 6 workers)."
echo "──────────────────────────────────────────────────────────────"
echo "Run Task 2 — Histogram (replace <window_size> with e.g. 8, 16, 32):"
echo ""
echo "  docker exec master bash -c \\"
echo "    'mpirun --allow-run-as-root --hostfile /app/hostfile -np 7 \\"
echo "     --mca btl tcp,self \\"
echo "     --mca btl_tcp_if_include 172.28.0.0/24,172.25.0.0/24 \\"
echo "     /app/task2_histogram_mpi /app/images <window_size> /app/output/histograms.json'"
echo ""
echo "Example — window size 8:"
echo "  docker exec master bash -c \\"
echo "    'mpirun --allow-run-as-root --hostfile /app/hostfile -np 7 \\"
echo "     --mca btl tcp,self \\"
echo "     --mca btl_tcp_if_include 172.28.0.0/24,172.25.0.0/24 \\"
echo "     /app/task2_histogram_mpi /app/images 8 /app/output/histograms.json'"
echo ""
echo "Output saved to: ./output/histograms.json"
echo "──────────────────────────────────────────────────────────────"
