# KISSKI Submit -- UnifoLM-VLA volles Finetuning

Deployt dieses Repo per Apptainer/SIF-Container auf der KISSKI-HPC und startet das UnifoLM-VLA-Training automatisiert. **Compute-Nodes haben kein Internet** (siehe [`../../Dokumentation/Training/kisski_hpc_ausweichen.md`](../../Dokumentation/Training/kisski_hpc_ausweichen.md)) -- daher passiert alles, was Netzwerk braucht (Image ziehen, Daten laden, Checkpoints abholen), **außerhalb** des Compute-Jobs: auf dieser Maschine bzw. dem KISSKI-Login-/Transfer-Node.

**KISSKI-spezifische Regeln** (bestätigt durch die aktuelle, datierte Doku eines Kollegen mit bereits produktivem KISSKI-Workflow -- zwei frühere Annahmen in diesem Ordner waren falsch und wurden korrigiert, siehe git-History):
- `apptainer pull` läuft **auf dem Login-Node** (`glogin-gpu.hpc.gwdg.de`), **nicht** per `srun` auf einem Compute-Node.
- **`/scratch/` wurde am 2026-03-31 abgeschaltet.** Alle Daten (Datensatz, Checkpoints) gehören auf den **VAST-Projekt-Storage** (`/mnt/vast-kisski/projects/<projekt>/`), nicht auf Scratch.
- Große Dateien (Datensatz, Checkpoints) laufen für Up-/Download trotzdem über den **Transfer-Node** (`transfer.hpc.gwdg.de`), nicht per `scp`/`rsync` direkt auf den Login-Node.
- **Geteilter Projekt-Storage mit einer anderen Gruppe** (z.B. `kisski-humrob`): Gruppe 1 (GR00T-Workflow) hat dort schon `data/`, `repo/`, `repo-groot/` liegen. Unsere Daten/Repos landen deshalb bewusst in einem eigenen Unterordner -- `KISSKI_GROUP_SUBDIR` (Default `gruppe2`) unter `KISSKI_PROJECT_DIR` -- statt direkt im gemeinsamen Wurzelverzeichnis.

**Status:** Docker-Image baut erfolgreich und wurde lokal verifiziert -- alle Kernpakete importieren fehlerfrei (`torch`, `flash_attn`, `unifolm_vla`, `lerobot`, `tensorflow`, `deepspeed`), inkl. echtem GPU-Zugriff (`docker run --gpus all` erkennt die lokale GPU korrekt über CUDA). Nach Docker Hub gepusht (`fichtlff/unifolm-vla-kisski:latest`). Ein Test-Job (`MAX_STEPS=10`) wurde bereits eingereicht (siehe [Aktueller Stand](#aktueller-stand-laufender-testlauf) unten) -- der eigentliche Trainingslauf ist noch nicht bestätigt erfolgreich. Mechanik (SLURM-Direktiven, Token-Dateien, Apptainer-Bind-Muster) ist aus dem bereits produktiv laufenden GR00T-Workflow eines Kollegen übernommen, aber auf unser eigenes Image + UnifoLM-VLA-Trainingskommando umgeschrieben.

## Aktueller Stand (laufender Testlauf)

Konkretes Setup für diesen Account (Stand 2026-09-06):
- **KISSKI-Projekt:** `kisski-humrob` (geteilt mit Gruppe 1) -- `KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/kisski-humrob`
- **Repo-Checkout auf dem Cluster:** `/user/fabian.fichtl/u29848/.project/dir.project/repo_2/Masterprojekt_G1_CubeStack` (NICHT unter `KISSKI_PROJECT_DIR` -- deshalb `REPO_DIR=...` bei jedem `sbatch` explizit mitgeben)
- **Login-Node:** `glogin9`/`glogin10` (mehrere Login-Knoten hinter `glogin-gpu.hpc.gwdg.de`)
- **Docker Hub Image:** `fichtlff/unifolm-vla-kisski:latest`

Checkliste, bevor ein Testlauf tatsächlich das Training erreicht:

| # | Voraussetzung | Status (Stand 2026-09-06) | Befehl |
|---|---|---|---|
| 1 | `logs/`-Verzeichnis existiert in `Training/Kisski_Submit/` (sonst kann SLURM keine Log-Datei anlegen und der Job stirbt sofort) | zu prüfen | `mkdir -p logs` |
| 2 | SIF-Image unter `$HOME/images/unifolm-vla-kisski.sif` | **fehlt noch** (`ls` bestätigt: nicht vorhanden) | `module load apptainer && apptainer pull $HOME/images/unifolm-vla-kisski.sif docker://fichtlff/unifolm-vla-kisski:latest` (auf dem Login-Node, ohne `srun`) |
| 3 | Datensatz + Basis-VLM unter `$KISSKI_PROJECT_DIR/gruppe2/data/` | **fehlt noch** (`prepare_and_stage_data.sh` noch nicht gelaufen) | `prepare_and_stage_data.sh` von einer Maschine mit Docker + Internet |

Job `15766148` (`MAX_STEPS=10`, `sbatch --export=ALL`) wurde eingereicht, bevor 2 und 3 erledigt waren -- er wird daher voraussichtlich mit einem Fehler zum fehlenden SIF-Image bzw. fehlenden Daten abbrechen, sobald er einen Knoten bekommt. Das ist erwartet und kein neuer Bug; einfach nach Beheben von 1-3 neu einreichen.

`HF_TOKEN`/`WANDB_API_KEY` sind für **diesen** Testlauf NICHT nötig: `SKIP_DOWNLOAD=1` ist der Default (kein Download im Job selbst), und `WANDB_API_KEY` ist bei uns (anders als beim Kollegen) optional.

## Ablauf

```
[diese Maschine]              [Docker Hub]                [Login-Node]              [Compute-Node]
                                                            glogin-gpu.hpc.gwdg.de    (KEIN Internet)
1. build_and_transfer_sif.sh ──push──► <namespace>/...
                                            │
                                            └── ssh ──► apptainer pull → $HOME/images/*.sif
2. prepare_and_stage_data.sh  ──rsync (über Transfer-Node)─► /mnt/vast-kisski/projects/<projekt>/gruppe2/data/
3. git clone (manuell, auf Login-Node -- git braucht Internet, das hat nur der Login-Node)
                                                              4. sbatch kisski_submit.sh ──►  apptainer run
                                                                                                (SKIP_DOWNLOAD=1,
                                                                                                 nur Konvertierung
                                                                                                 + Training)
5. fetch_checkpoints.sh       ◄──rsync (über Transfer-Node)── /mnt/vast-kisski/projects/<projekt>/gruppe2/data/outputs_unifolm_vla/
   (+ optional Push nach HF)
```

## 1. Image bauen + über Docker Hub auf KISSKI ziehen

Kein `scp` eines mehrere GB großen Tarballs nötig -- Image wird lokal gebaut, zu Docker Hub gepusht (braucht vorher `docker login`), und auf dem **Login-Node** per `apptainer pull docker://...` direkt gezogen. Das ist exakt das Muster, das im GR00T-Workflow des Kollegen bereits nachweislich produktiv läuft:

```bash
DOCKERHUB_REPO=<dein-dockerhub-username>/unifolm-vla-kisski \
KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de \
./build_and_transfer_sif.sh
```

`IMAGE_TAG` (Default: `latest`) ist bei Bedarf überschreibbar. Ist das Image schon gebaut/gepusht, sparen `SKIP_BUILD=1`/`SKIP_PUSH=1` die entsprechenden Schritte.

## 2. Daten vorbereiten + hochladen

Lädt Basis-VLM + Datensatz von Hugging Face und konvertiert sie (LeRobot → HDF5 → RLDS) -- lokal, mit Internetzugang, über denselben `entrypoint.sh` wie später der Cluster-Job (nur via `docker run` statt `apptainer run`, damit die schweren Python-Abhängigkeiten nicht auf dem Host installiert sein müssen). Landet auf dem VAST-Projekt-Storage, transferiert über den Transfer-Node:

```bash
KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de \
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
HF_TOKEN=hf_... \
./prepare_and_stage_data.sh
```

## 3. Repo auf KISSKI klonen

Auf dem **Login-Node** (der hat Internet, der Compute-Node nicht):

```bash
git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git \
    /mnt/vast-kisski/projects/<projekt>/gruppe2/repo
```

## 4. Tokens hinterlegen + Job einreichen

```bash
printf 'hf_DEIN_TOKEN\n'  > ~/.hf_token  && chmod 600 ~/.hf_token   # nur falls SKIP_DOWNLOAD=0 gebraucht wird
printf 'DEIN_WANDB_KEY\n' > ~/.wandb_key && chmod 600 ~/.wandb_key  # optional
cd /mnt/vast-kisski/projects/<projekt>/gruppe2/repo/Training/Kisski_Submit
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
sbatch --export=ALL kisski_submit.sh
```

Volles Finetuning (Backbone nicht eingefroren) ist der Default (`FREEZE_BACKBONE=0`). Für einen Action-Head-only-Vergleichslauf mit größerem Batch als lokal:

```bash
FREEZE_BACKBONE=1 KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> sbatch --export=ALL kisski_submit.sh
```

**Wichtig:** Env-Variablen vor `sbatch` wirken nur mit explizitem `--export=ALL` -- KISSKI setzt `SBATCH_EXPORT=none` auf dem Login-Node, das überstimmt sonst die `#SBATCH --export=ALL`-Direktive im Skript.

### Erster Testlauf (empfohlen vor einem echten 48h-Job)

```bash
MAX_STEPS=10 KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> sbatch --export=ALL kisski_submit.sh
```

## 5. Checkpoints zurückholen

Checkpoints landen nur auf dem VAST-Projekt-Storage -- der Compute-Node kann sie nicht selbst irgendwohin pushen (kein Internet). Abholen von dieser Maschine aus, über den Transfer-Node:

```bash
KISSKI_LOGIN_HOST=<username>@glogin-gpu.hpc.gwdg.de \
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
RUN_ID=g1_dex3_blockstacking_full \
./fetch_checkpoints.sh

# optional direkt mit Push nach Hugging Face:
HF_UPLOAD_REPO=<namespace>/unifolm-vla-g1-dex3-full ./fetch_checkpoints.sh
```

## Dateien

| Datei | Zweck |
|---|---|
| [`../UnifoLM-VLA/Dockerfile`](../UnifoLM-VLA/Dockerfile) | Baut das Image (Python-Env + UnifoLM-VLA installiert). Skripte werden NICHT eingebacken, sondern zur Laufzeit gemountet. |
| [`build_and_transfer_sif.sh`](build_and_transfer_sif.sh) | Lokal bauen → Docker-Hub-Push → `apptainer pull` auf dem Login-Node. |
| [`prepare_and_stage_data.sh`](prepare_and_stage_data.sh) | Lokal Daten laden + konvertieren → rsync (über Transfer-Node) auf VAST-Projekt-Storage. |
| [`entrypoint.sh`](entrypoint.sh) | Container-Entrypoint (Download → Konvertierung → Training), läuft sowohl lokal (Docker) als auch im Cluster-Job (Apptainer). |
| [`kisski_submit.sh`](kisski_submit.sh) | SLURM-Batch-Skript (`sbatch kisski_submit.sh`), startet den Container mit den richtigen Binds/Env-Variablen. `SKIP_DOWNLOAD=1` per Default. |
| [`fetch_checkpoints.sh`](fetch_checkpoints.sh) | Checkpoints per rsync (über Transfer-Node) zurückholen, optional Push nach Hugging Face. |

## Bereits verifiziert

- `docker build` läuft vollständig durch (Dockerfile + `requirements-frozen.txt`), ca. 25-30 Minuten (fast ausschließlich `flash_attn`-Kompilierung).
- `torch`, `flash_attn`, `unifolm_vla`, `lerobot`, `tensorflow`, `deepspeed` importieren fehlerfrei im gebauten Image.
- `docker run --gpus all` erkennt die GPU korrekt (`torch.cuda.is_available() == True`, `torch.cuda.get_device_name(0)` liefert die echte Karte).
- Image erfolgreich zu Docker Hub gepusht (`fichtlff/unifolm-vla-kisski:latest`).

## Offene Punkte / bekannte Unsicherheiten

- `entrypoint.sh` patched die Konstante `HDF5_DATA_DIR` in `rlds_dataset_g1_dex3.py` per `sed` auf den Container-Pfad (die Datei liest den Pfad nicht aus einer Env-Variable) -- funktional plausibel, aber ungetestet.
- `UnifoLM-VLM-Base`-Downloadpfad (`unitreerobotics/UnifoLM-VLM-Base`) ist nicht gegen einen echten HF-Repo-Namen verifiziert -- ggf. anpassen.
- Ressourcen-Direktiven (`-G A100:4`, `--mem=384G`, `-t 48:00:00`) sind vom GR00T-Lauf des Kollegen übernommen und müssen ggf. für UnifoLM-VLA neu kalibriert werden (anderes Modell, andere Speicherprofile).
- `apptainer pull docker://...` auf dem Login-Node läuft gerade zum ersten Mal gegen echte Hardware (Stand 2026-09-06 noch nicht als abgeschlossen bestätigt) -- damit auch der erste echte Test, ob der Login-Node tatsächlich Internetzugriff auf Docker Hub hat.
- Der eigentliche Trainingslauf (`entrypoint.sh` Stufe 3, `accelerate launch ... train_unifolm_vla.py`) wurde in diesem Image noch nicht ausgeführt, nur die Imports verifiziert.
- Aktueller Detailstatus + Checkliste: [Aktueller Stand](#aktueller-stand-laufender-testlauf) oben.
