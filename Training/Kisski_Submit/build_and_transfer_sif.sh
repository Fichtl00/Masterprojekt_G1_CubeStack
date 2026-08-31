#!/usr/bin/env bash
# build_and_transfer_sif.sh -- Baut das UnifoLM-VLA-Trainingsimage LOKAL (auf
# dieser Maschine, mit Docker) und konvertiert/transferiert es DIREKT als SIF
# auf den KISSKI-Login-Node -- ohne Umweg über eine Docker-Registry (kein Docker
# Hub o.ä. nötig, kein öffentliches Image).
#
# Voraussetzung: Apptainer/Singularity auf dem Login-Node kann ein SIF direkt aus
# einem "docker-archive" (= `docker save`-Tarball) bauen -- Standard-Feature,
# braucht selbst keine Internetverbindung auf dem Cluster.
#
# Usage:
#   KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
#   KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
#   ./build_and_transfer_sif.sh
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_HOST="${KISSKI_HOST:?Setze KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de}"
KISSKI_PROJECT_DIR="${KISSKI_PROJECT_DIR:-/mnt/vast-kisski/projects/kisski-humrob}"
IMAGE_NAME="${IMAGE_NAME:-unifolm-vla-kisski:latest}"
TAR_PATH="${TAR_PATH:-/tmp/unifolm-vla-kisski.tar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_DIR="${SCRIPT_DIR}/../UnifoLM-VLA"

echo "==> 1/4 Baue Docker-Image lokal ($IMAGE_NAME) ..."
docker build -t "$IMAGE_NAME" -f "${DOCKERFILE_DIR}/Dockerfile" "$DOCKERFILE_DIR"

echo "==> 2/4 Exportiere Image als Tarball ($TAR_PATH) ..."
docker save -o "$TAR_PATH" "$IMAGE_NAME"
du -h "$TAR_PATH"

echo "==> 3/4 Transferiere Tarball nach $KISSKI_HOST:${KISSKI_PROJECT_DIR}/images/ ..."
ssh "$KISSKI_HOST" "mkdir -p ${KISSKI_PROJECT_DIR}/images \$HOME/images"
scp "$TAR_PATH" "${KISSKI_HOST}:${KISSKI_PROJECT_DIR}/images/unifolm-vla-kisski.tar"

echo "==> 4/4 Baue SIF auf dem Login-Node (apptainer build aus docker-archive) ..."
ssh "$KISSKI_HOST" bash -s << EOF
set -euo pipefail
module load apptainer
apptainer build \$HOME/images/unifolm-vla-kisski.sif \
    docker-archive://${KISSKI_PROJECT_DIR}/images/unifolm-vla-kisski.tar
echo "SIF erstellt: \$HOME/images/unifolm-vla-kisski.sif"
EOF

echo "==> Fertig. Lokalen Tarball aufräumen: rm $TAR_PATH"
