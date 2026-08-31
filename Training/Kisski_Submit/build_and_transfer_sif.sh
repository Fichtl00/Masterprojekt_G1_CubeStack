#!/usr/bin/env bash
# build_and_transfer_sif.sh -- Baut das UnifoLM-VLA-Trainingsimage LOKAL (auf
# dieser Maschine, mit Docker), pusht es zu Docker Hub, und zieht es auf KISSKI
# direkt von dort als SIF -- kein scp/docker-archive-Tarball noetig.
#
# Warum Docker Hub statt scp+docker-archive: die Session, in der dieses Repo
# entstand, hatte keinen Netzwerkpfad zu KISSKI (glogin10 nicht aufloesbar --
# vermutlich GWDG-VPN-only) und daher kein Geraet zur Hand, das gleichzeitig
# Docker UND einen Pfad zu KISSKI hat. Docker-Hub-Push lief von hier aus
# problemlos; ein `apptainer pull docker://...` direkt auf KISSKI ist ausserdem
# GENAU das Muster, das im GR00T-Workflow eines Kollegen bereits nachweislich
# funktioniert (siehe dortiges kisski_submit.sh).
#
# WICHTIGE KISSKI-SPEZIFIKA (vom GWDG-Support/Kollegen bestätigt):
#  - `apptainer pull` darf NICHT auf dem Login-Node laufen (Ressourcen-Policy)
#    -- wird hier per `srun` auf einem Compute-Node ausgefuehrt.
#  - Die fertige .sif-Datei ist klein genug fuer $HOME/images/.
#
# Usage:
#   DOCKERHUB_REPO=<dein-dockerhub-username>/unifolm-vla-kisski \
#   KISSKI_LOGIN_HOST=<username>@glogin10 \
#   ./build_and_transfer_sif.sh
#
#   # Image schon gebaut + gepusht? Baue nur den Docker-Build-Schritt aus:
#   SKIP_BUILD=1 SKIP_PUSH=1 DOCKERHUB_REPO=... KISSKI_LOGIN_HOST=... ./build_and_transfer_sif.sh
#
# NOCH NICHT gegen einen echten KISSKI-Login-Node getestet (der Docker-Hub-Push
# selbst wurde bereits erfolgreich durchgefuehrt).

set -euo pipefail

DOCKERHUB_REPO="${DOCKERHUB_REPO:?Setze DOCKERHUB_REPO=<dein-dockerhub-username>/unifolm-vla-kisski}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
KISSKI_LOGIN_HOST="${KISSKI_LOGIN_HOST:?Setze KISSKI_LOGIN_HOST=<username>@glogin10}"
KISSKI_PARTITION="${KISSKI_PARTITION:-kisski}"
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

echo "==> 3/3 Ziehe Image als SIF auf einem COMPUTE-Node (per srun, nicht auf dem Login-Node) ..."
ssh "$KISSKI_LOGIN_HOST" bash -s << EOF
set -euo pipefail
mkdir -p \$HOME/images
srun -p "${KISSKI_PARTITION}" -t 00:30:00 -c 4 --mem=16G bash -c '
    module load apptainer
    apptainer pull \$HOME/images/unifolm-vla-kisski.sif docker://${FULL_IMAGE}
'
echo "SIF erstellt: \$HOME/images/unifolm-vla-kisski.sif"
EOF

echo "==> Fertig."
