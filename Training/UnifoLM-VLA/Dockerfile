# Build-Kontext: Training/UnifoLM-VLA/ (dieses Verzeichnis, enthaelt pyproject.toml)
#   docker build -t <namespace>/unifolm-vla-kisski:latest -f Dockerfile .
#
# Analog zum Vorgehen des Kollegen (lucam03/projekt-humanoider-roboter fuer GR00T):
# fertiges Image enthaelt nur die Python-Umgebung. Die eigentlichen Trainings-/
# Entrypoint-Skripte werden zur Laufzeit aus dem Repo in den Container gemountet
# (--bind .../Training/Kisski_Submit:/scripts), damit Aenderungen ohne Image-
# Rebuild wirken -- siehe kisski_submit.sh.
# nvidia/cuda:...-devel (nicht pytorch/pytorch, nicht -runtime):
#  - devel statt runtime: flash_attn==2.5.6 hat kein passendes Prebuilt-Wheel fuer
#    diese Torch/CUDA/Python-Kombination und muss aus dem Quellcode gebaut werden --
#    dafuer wird nvcc/CUDA_HOME benoetigt.
#  - ubuntu22.04 statt pytorch/pytorch-Image: requirements-frozen.txt ist ein
#    `pip freeze` aus einer nachweislich funktionierenden lokalen Python-3.10-Umgebung
#    (Training in diesem Projekt bereits erfolgreich damit gelaufen). pytorch/pytorch-
#    Images bringen Python 3.11 mit, wofuer einzelne exakt gepinnte Pakete (z.B.
#    tensorflow-addons==0.19.0) kein Wheel haben. Ubuntu 22.04 hat Python 3.10 als
#    System-Default -- passt exakt.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev git ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3
ENV CUDA_HOME=/usr/local/cuda

# Ubuntu 22.04's apt-pip (22.0.2) + setuptools (59.6.0) parsen PEP-621-Metadaten
# (pyproject.toml [project]-Tabelle) fehlerhaft -- git-Abhaengigkeiten wie `lerobot`
# bauen dann mit Namen "UNKNOWN" statt "lerobot" ("Fix your #egg=... fragments").
# Erst pip/setuptools/wheel aktualisieren behebt das.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app/unifolm-vla
COPY . /app/unifolm-vla

# pyproject.toml hat einen echten, ungeloesten Konflikt (tensorflow_graphics ->
# tensorflow-addons braucht typeguard<3.0, tyro braucht typeguard>=4.0) --
# `pip install -e .` mit normaler Resolution schlaegt fehl (ResolutionImpossible).
# requirements-frozen.txt ist ein `pip freeze` aus einer nachweislich funktionierenden
# lokalen UnifoLM-VLA-Umgebung (Training in diesem Projekt bereits erfolgreich damit
# gelaufen) -- installiert mit --no-deps, um Pips Resolver komplett zu umgehen.
# evdev/pynput waren im Freeze enthalten (Teil derselben Conda-Umgebung fuer
# Teleop-Tools), werden von UnifoLM-VLA selbst aber nirgends importiert und lassen
# sich in diesem Image nicht bauen (fehlende Linux-Input-Header) -- entfernt.
COPY requirements-frozen.txt /tmp/requirements-frozen.txt

# flash_attn's setup.py importiert `packaging`/`torch` direkt (kein deklariertes
# Build-Requirement) -- braucht --no-build-isolation + vorab installierte torch/
# packaging/ninja im Ziel-Environment, sonst schlaegt die Metadaten-Generierung fehl.
RUN pip install --no-cache-dir packaging==26.0 ninja==1.13.0 torch==2.5.1
RUN grep -v '^flash_attn==' /tmp/requirements-frozen.txt > /tmp/requirements-no-flash.txt \
    && pip install --no-cache-dir --no-deps -r /tmp/requirements-no-flash.txt \
    && pip install --no-cache-dir --no-deps --no-build-isolation flash_attn==2.5.6 \
    && pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["/scripts/entrypoint.sh"]
