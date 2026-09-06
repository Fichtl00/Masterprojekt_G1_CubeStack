#!/usr/bin/env bash
# build_and_transfer_sif.sh -- Baut das UnifoLM-VLA-Trainingsimage LOKAL (auf
# dieser Maschine, mit Docker), pusht es zu Docker Hub, und zieht es auf KISSKI
# direkt von dort als SIF -- kein scp/docker-archive-Tarball noetig.
#
# WICHTIG (bestaetigt durch die aktuelle, datierte KISSKI-Doku eines Kollegen mit
# bereits produktivem Workflow -- korrigiert gegenueber einer frueheren Version
# dieses Skripts, siehe git-History):
#  - `apptainer pull` laeuft direkt auf dem LOGIN-Node (glogin-gpu.hpc.gwdg.de),
#    NICHT per `srun` auf einem Compute-Node. Frueher lief hier ein `srun`-Wrapper
#    (auf Basis einer anderen, offenbar veralteten Kollegen-Notiz) -- das fuehrte
#    zu Timeouts, weil die Compute-Nodes tatsaechlich kein Internet haben.
#  - Warum Docker Hub statt scp+docker-archive: die Session, in der dieses Skript
#    entstand, hatte keinen Netzwerkpfad zu KISSKI und kein Geraet mit Docker UND
#    Cluster-Zugang zugleich. Ein Docker-Hub-Push lief problemlos, und
#    `apptainer pull docker://...` auf dem Login-Node ist ausserdem GENAU das
#    Muster, das im GR00T-Workflow des Kollegen bereits produktiv laeuft.
#
# Usage:
#   DOCKERHUB_REPO=<dein-dockerhub-username>/unifolm-vla-kisski \
#   KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de \
#   ./build_and_transfer_sif.sh
#
#   # Image schon gebaut + gepusht? Baue nur den Docker-Build-Schritt aus:
#   SKIP_BUILD=1 SKIP_PUSH=1 DOCKERHUB_REPO=... KISSKI_LOGIN_HOST=... ./build_and_transfer_sif.sh

set -euo pipefail

DOCKERHUB_REPO="${DOCKERHUB_REPO:?Setze DOCKERHUB_REPO=<dein-dockerhub-username>/unifolm-vla-kisski}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_PUSH="${SKIP_PUSH:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE_DIR="${SCRIPT_DIR}/../UnifoLM-VLA"
FULL_IMAGE="${DOCKERHUB_REPO}:${IMAGE_TAG}"

if [[ "$SKIP_BUILD" != "1" ]]; then
    echo "==> 1/3 Baue Docker-Image lokal ($FULL_IMAGE) ..."
    docker build -t "$FULL_IMAGE" -f "${DOCKERFILE_DIR}/Dockerfile" "$DOCKERFILE_DIR"
else
    echo "==> 1/3 SKIP_BUILD=1, übersprungen."
fi

if [[ "$SKIP_PUSH" != "1" ]]; then
    echo "==> 2/3 Push nach Docker Hub ($FULL_IMAGE) -- 'docker login' muss vorher erfolgt sein ..."
    docker push "$FULL_IMAGE"
else
    echo "==> 2/3 SKIP_PUSH=1, übersprungen."
fi

echo "==> 3/3 Ziehe Image als SIF auf dem LOGIN-Node ..."
ssh "$KISSKI_LOGIN_HOST" bash -s << EOF
set -euo pipefail
module load apptainer
mkdir -p \$HOME/images
apptainer pull \$HOME/images/unifolm-vla-kisski.sif docker://${FULL_IMAGE}
echo "SIF erstellt: \$HOME/images/unifolm-vla-kisski.sif"
EOF

echo "==> Fertig."
