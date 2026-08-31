#!/usr/bin/env bash
# build_and_transfer_sif.sh -- Baut das UnifoLM-VLA-Trainingsimage LOKAL (auf
# dieser Maschine, mit Docker) und konvertiert/transferiert es DIREKT als SIF
# auf KISSKI -- ohne Umweg über eine Docker-Registry (kein Docker Hub o.ä.
# nötig, kein öffentliches Image).
#
# WICHTIGE KISSKI-SPEZIFIKA (vom GWDG-Support/Kollegen bestätigt):
#  - `apptainer build`/`pull` darf NICHT auf dem Login-Node laufen (Ressourcen-
#    Policy) -- wird hier per `srun` auf einem Compute-Node ausgefuehrt.
#  - Grosse Dateien (der Image-Tarball) gehoeren auf den Transfer-Node
#    (transfer.hpc.gwdg.de), nicht per scp direkt auf den Login-Node.
#  - Die fertige .sif-Datei ist klein genug fuer $HOME/images/ (wie im
#    GWDG-Beispiel).
#
# Usage:
#   KISSKI_LOGIN_HOST=<username>@glogin10 \
#   ./build_and_transfer_sif.sh
#
#   # Optional, falls abweichend von den Defaults:
#   KISSKI_TRANSFER_HOST=<username>@transfer.hpc.gwdg.de
#   KISSKI_SCRATCH_DIR=/scratch/<username>
#   KISSKI_PARTITION=kisski   (oder kisski-h100)
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet.

set -euo pipefail

KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin10}"
KISSKI_USER="${KISSKI_LOGIN_HOST%@*}"
KISSKI_HOSTNAME_ONLY="${KISSKI_LOGIN_HOST#*@}"
_GWDG_SUFFIX=".hpc.gwdg.de"
KISSKI_TRANSFER_HOST="${KISSKI_TRANSFER_HOST:-${KISSKI_USER}@transfer${_GWDG_SUFFIX}}"
KISSKI_SCRATCH_DIR="${KISSKI_SCRATCH_DIR:-/scratch/${KISSKI_USER}}"
KISSKI_PARTITION="${KISSKI_PARTITION:-kisski}"
IMAGE_NAME="${IMAGE_NAME:-unifolm-vla-kisski:latest}"
TAR_PATH="${TAR_PATH:-/tmp/unifolm-vla-kisski.tar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_DIR="${SCRIPT_DIR}/../UnifoLM-VLA"

echo "==> 1/5 Baue Docker-Image lokal ($IMAGE_NAME) ..."
docker build -t "$IMAGE_NAME" -f "${DOCKERFILE_DIR}/Dockerfile" "$DOCKERFILE_DIR"

echo "==> 2/5 Exportiere Image als Tarball ($TAR_PATH) ..."
docker save -o "$TAR_PATH" "$IMAGE_NAME"
du -h "$TAR_PATH"

echo "==> 3/5 Transferiere Tarball zum Transfer-Node ($KISSKI_TRANSFER_HOST, nicht Login-Node) ..."
ssh "$KISSKI_TRANSFER_HOST" "mkdir -p ${KISSKI_SCRATCH_DIR}/images"
scp "$TAR_PATH" "${KISSKI_TRANSFER_HOST}:${KISSKI_SCRATCH_DIR}/images/unifolm-vla-kisski.tar"

echo "==> 4/5 Baue SIF auf einem COMPUTE-Node (per srun, nicht auf dem Login-Node) ..."
ssh "$KISSKI_LOGIN_HOST" bash -s << EOF
set -euo pipefail
mkdir -p \$HOME/images
srun -p "${KISSKI_PARTITION}" -t 00:30:00 -c 4 --mem=16G bash -c '
    module load apptainer
    apptainer build \$HOME/images/unifolm-vla-kisski.sif \
        docker-archive://${KISSKI_SCRATCH_DIR}/images/unifolm-vla-kisski.tar
'
echo "SIF erstellt: \$HOME/images/unifolm-vla-kisski.sif"
EOF

echo "==> 5/5 Aufräumen: Tarball auf Scratch + lokal loeschen ..."
ssh "$KISSKI_TRANSFER_HOST" "rm -f ${KISSKI_SCRATCH_DIR}/images/unifolm-vla-kisski.tar"
echo "    Lokal noch manuell aufräumen: rm $TAR_PATH"
