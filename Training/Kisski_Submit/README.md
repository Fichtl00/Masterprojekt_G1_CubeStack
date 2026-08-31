# KISSKI Submit -- UnifoLM-VLA volles Finetuning

Deployt dieses Repo per Apptainer/SIF-Container auf der KISSKI-HPC und startet das UnifoLM-VLA-Training automatisiert. **Compute-Nodes haben kein Internet** (siehe [`../../Dokumentation/Training/kisski_hpc_ausweichen.md`](../../Dokumentation/Training/kisski_hpc_ausweichen.md)) -- daher passiert alles, was Netzwerk braucht (Image bauen, Daten laden, Checkpoints abholen), **außerhalb** des Compute-Jobs: auf dieser Maschine bzw. dem KISSKI-Login-/Transfer-Node.

**KISSKI-spezifische Regeln, die eingehalten werden:**
- `apptainer build`/`pull` darf **nicht auf dem Login-Node** laufen (Ressourcen-Policy) -- läuft hier per `srun` auf einem Compute-Node.
- Große Dateien (Image-Tarball, Datensatz, Checkpoints) laufen über den **Transfer-Node** (`transfer.hpc.gwdg.de`), nicht per `scp`/`rsync` direkt auf den Login-Node.
- Große Daten gehören auf **`/scratch/$USER`**, nicht in `$HOME` oder ein Projektverzeichnis. Die SIF-Datei selbst (klein) liegt in `$HOME/images/`.

**Status:** Docker-Image baut erfolgreich und wurde lokal verifiziert -- alle Kernpakete importieren fehlerfrei (`torch`, `flash_attn`, `unifolm_vla`, `lerobot`, `tensorflow`, `deepspeed`), inkl. echtem GPU-Zugriff (`docker run --gpus all` erkennt die lokale GPU korrekt über CUDA). **Noch nicht auf echter KISSKI-Hardware getestet** -- SLURM-Direktiven, die `srun`-gestützte Apptainer-Konvertierung und der eigentliche Trainingslauf sind ungetestet. Mechanik (SLURM-Direktiven, Token-Dateien, Apptainer-Bind-Muster) ist aus dem bereits produktiv laufenden GR00T-Workflow eines Kollegen übernommen, aber auf unser eigenes Image + UnifoLM-VLA-Trainingskommando umgeschrieben -- vor dem ersten echten Lauf unbedingt mit kleinen `MAX_STEPS`-Werten verifizieren.

## Ablauf

```
[diese Maschine]                    [Transfer-Node]        [Login-Node]           [Compute-Node]
                                     transfer.hpc.gwdg.de   glogin10               (KEIN Internet)
1. build_and_transfer_sif.sh  ──scp──►  /scratch/$USER/images/*.tar
                                                              │
                                                              └─ srun ──►  apptainer build → $HOME/images/*.sif
2. prepare_and_stage_data.sh  ──rsync─► /scratch/$USER/data/ (Basis-VLM + RLDS-Datensatz)
3. git clone (manuell, auf Login-Node -- git braucht Internet, das hat nur der Login-Node)
                                                              4. sbatch kisski_submit.sh ──►  apptainer run
                                                                                                (SKIP_DOWNLOAD=1,
                                                                                                 nur Konvertierung
                                                                                                 + Training)
5. fetch_checkpoints.sh       ◄──rsync── /scratch/$USER/data/outputs_unifolm_vla/
   (+ optional Push nach HF)
```

## 1. Image bauen + als SIF transferieren

Kein Docker Hub nötig -- Image wird lokal gebaut, als Tarball exportiert, über den Transfer-Node hochgeladen und per `srun` auf einem Compute-Node (nicht dem Login-Node!) zu einem SIF konvertiert:

```bash
KISSKI_LOGIN_HOST=<username>@glogin10 \
./build_and_transfer_sif.sh
```

`KISSKI_TRANSFER_HOST` (Default: `<username>@transfer.hpc.gwdg.de`), `KISSKI_SCRATCH_DIR` (Default: `/scratch/<username>`) und `KISSKI_PARTITION` (Default: `kisski`) sind bei Bedarf überschreibbar.

## 2. Daten vorbereiten + hochladen

Lädt Basis-VLM + Datensatz von Hugging Face und konvertiert sie (LeRobot → HDF5 → RLDS) -- lokal, mit Internetzugang, über denselben `entrypoint.sh` wie später der Cluster-Job (nur via `docker run` statt `apptainer run`, damit die schweren Python-Abhängigkeiten nicht auf dem Host installiert sein müssen). Landet auf `/scratch/$USER/data`, transferiert über den Transfer-Node:

```bash
KISSKI_LOGIN_HOST=<username>@glogin10 \
HF_TOKEN=hf_... \
./prepare_and_stage_data.sh
```

## 3. Repo auf KISSKI klonen

Auf dem **Login-Node** (der hat Internet, der Compute-Node nicht). Kleine Datei-Menge (nur Code) -- `$HOME` oder ein Projektverzeichnis sind hierfür in Ordnung:

```bash
git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git $HOME/repo
```

Liegt das Repo woanders (z.B. in einem Projektverzeichnis), `REPO_DIR=...` beim `sbatch`-Aufruf in Schritt 4 entsprechend setzen.

## 4. Tokens hinterlegen + Job einreichen

```bash
printf 'hf_DEIN_TOKEN\n'  > ~/.hf_token  && chmod 600 ~/.hf_token   # nur falls SKIP_DOWNLOAD=0 gebraucht wird
printf 'DEIN_WANDB_KEY\n' > ~/.wandb_key && chmod 600 ~/.wandb_key  # optional
cd $HOME/repo/Training/Kisski_Submit   # oder wo auch immer REPO_DIR liegt
sbatch kisski_submit.sh
```

Volles Finetuning (Backbone nicht eingefroren) ist der Default (`FREEZE_BACKBONE=0`). Für einen Action-Head-only-Vergleichslauf mit größerem Batch als lokal:

```bash
FREEZE_BACKBONE=1 sbatch --export=ALL kisski_submit.sh
```

**Wichtig:** Env-Variablen vor `sbatch` wirken nur mit explizitem `--export=ALL` -- KISSKI setzt `SBATCH_EXPORT=none` auf dem Login-Node, das überstimmt sonst die `#SBATCH --export=ALL`-Direktive im Skript. Weicht `REPO_DIR` vom Default (`$HOME/repo`) ab, muss es ebenfalls per `--export=ALL` mitgegeben werden:

```bash
REPO_DIR=/pfad/zu/deinem/repo sbatch --export=ALL kisski_submit.sh
```

### Erster Testlauf (empfohlen vor einem echten 48h-Job)

```bash
MAX_STEPS=10 sbatch --export=ALL kisski_submit.sh
```

## 5. Checkpoints zurückholen

Checkpoints landen nur auf `/scratch/$USER` -- der Compute-Node kann sie nicht selbst irgendwohin pushen (kein Internet). Abholen von dieser Maschine aus, über den Transfer-Node:

```bash
KISSKI_LOGIN_HOST=<username>@glogin10 \
RUN_ID=g1_dex3_blockstacking_full \
./fetch_checkpoints.sh

# optional direkt mit Push nach Hugging Face:
HF_UPLOAD_REPO=<namespace>/unifolm-vla-g1-dex3-full ./fetch_checkpoints.sh
```

## Dateien

| Datei | Zweck |
|---|---|
| [`../UnifoLM-VLA/Dockerfile`](../UnifoLM-VLA/Dockerfile) | Baut das Image (Python-Env + UnifoLM-VLA installiert). Skripte werden NICHT eingebacken, sondern zur Laufzeit gemountet. |
| [`build_and_transfer_sif.sh`](build_and_transfer_sif.sh) | Lokal bauen → Tarball → Transfer-Node → `srun` + `apptainer build` auf einem Compute-Node. |
| [`prepare_and_stage_data.sh`](prepare_and_stage_data.sh) | Lokal Daten laden + konvertieren → rsync (über Transfer-Node) nach `/scratch/$USER/data`. |
| [`entrypoint.sh`](entrypoint.sh) | Container-Entrypoint (Download → Konvertierung → Training), läuft sowohl lokal (Docker) als auch im Cluster-Job (Apptainer). |
| [`kisski_submit.sh`](kisski_submit.sh) | SLURM-Batch-Skript (`sbatch kisski_submit.sh`), startet den Container mit den richtigen Binds/Env-Variablen. `SKIP_DOWNLOAD=1` per Default. |
| [`fetch_checkpoints.sh`](fetch_checkpoints.sh) | Checkpoints per rsync (über Transfer-Node) zurückholen, optional Push nach Hugging Face. |

## Bereits verifiziert

- `docker build` läuft vollständig durch (Dockerfile + `requirements-frozen.txt`), ca. 25-30 Minuten (fast ausschließlich `flash_attn`-Kompilierung).
- `torch`, `flash_attn`, `unifolm_vla`, `lerobot`, `tensorflow`, `deepspeed` importieren fehlerfrei im gebauten Image.
- `docker run --gpus all` erkennt die GPU korrekt (`torch.cuda.is_available() == True`, `torch.cuda.get_device_name(0)` liefert die echte Karte).

## Offene Punkte / bekannte Unsicherheiten

- `entrypoint.sh` patched die Konstante `HDF5_DATA_DIR` in `rlds_dataset_g1_dex3.py` per `sed` auf den Container-Pfad (die Datei liest den Pfad nicht aus einer Env-Variable) -- funktional plausibel, aber ungetestet.
- `UnifoLM-VLM-Base`-Downloadpfad (`unitreerobotics/UnifoLM-VLM-Base`) ist nicht gegen einen echten HF-Repo-Namen verifiziert -- ggf. anpassen.
- Ressourcen-Direktiven (`-G A100:4`, `--mem=384G`, `-t 48:00:00`) sind vom GR00T-Lauf des Kollegen übernommen und müssen ggf. für UnifoLM-VLA neu kalibriert werden (anderes Modell, andere Speicherprofile).
- `KISSKI_PARTITION`/`srun`-Ressourcen für den SIF-Build (`-t 00:30:00 -c 4 --mem=16G`) sind eine Schätzung, nicht gegen echte KISSKI-Limits verifiziert.
- `apptainer build ... docker-archive://...` (Konvertierung des `docker save`-Tarballs zu SIF) ist ungetestet -- kein Apptainer auf dieser Maschine verfügbar.
- Der eigentliche Trainingslauf (`entrypoint.sh` Stufe 3, `accelerate launch ... train_unifolm_vla.py`) wurde in diesem Image noch nicht ausgeführt, nur die Imports verifiziert.
