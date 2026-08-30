# Build-Kontext: Training/UnifoLM-VLA/ (dieses Verzeichnis, enthaelt pyproject.toml)
#   docker build -t <namespace>/unifolm-vla-kisski:latest -f Dockerfile .
#
# Analog zum Vorgehen des Kollegen (lucam03/projekt-humanoider-roboter fuer GR00T):
# fertiges Image enthaelt nur die Python-Umgebung. Die eigentlichen Trainings-/
# Entrypoint-Skripte werden zur Laufzeit aus dem Repo in den Container gemountet
# (--bind .../Training/Kisski_Submit:/scripts), damit Aenderungen ohne Image-
# Rebuild wirken -- siehe kisski_submit.sh.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/unifolm-vla
COPY . /app/unifolm-vla
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["/scripts/entrypoint.sh"]
