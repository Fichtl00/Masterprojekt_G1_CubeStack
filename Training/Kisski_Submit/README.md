# KISSKI Submit -- UnifoLM-VLA volles Finetuning

Deployt dieses Repo per Apptainer/SIF-Container auf der KISSKI-HPC und startet das UnifoLM-VLA-Training automatisiert. **Compute-Nodes haben kein Internet** (siehe [`../../Dokumentation/Training/kisski_hpc_ausweichen.md`](../../Dokumentation/Training/kisski_hpc_ausweichen.md)) -- daher passiert alles, was Netzwerk braucht (Image bauen, Daten laden, Checkpoints abholen), **außerhalb** des Compute-Jobs: auf dieser Maschine bzw. dem KISSKI-Login-Node.

**Status: noch nicht auf echter KISSKI-Hardware getestet.** Mechanik (SLURM-Direktiven, Token-Dateien, Apptainer-Bind-Muster) ist aus dem bereits produktiv laufenden GR00T-Workflow eines Kollegen übernommen, aber auf unser eigenes Image + UnifoLM-VLA-Trainingskommando umgeschrieben -- vor dem ersten echten Lauf unbedingt mit kleinen `MAX_STEPS`-Werten verifizieren.

## Ablauf

```
[diese Maschine]                          [KISSKI-Login-Node]              [KISSKI-Compute-Node]
                                                                            (KEIN Internet)
1. build_and_transfer_sif.sh   ──scp──►   apptainer build → .sif
2. prepare_and_stage_data.sh   ──rsync─►  Basis-VLM + RLDS-Datensatz
                                            auf Projektspeicher
3. git clone (auf Login-Node, manuell)
                                           4. sbatch kisski_submit.sh ──────► apptainer run
                                                                              (SKIP_DOWNLOAD=1,
                                                                               nur Konvertierung
                                                                               + Training)
5. fetch_checkpoints.sh        ◄──rsync──  Checkpoints vom Projektspeicher
   (+ optional Push nach HF)
```

## 1. Image bauen + als SIF transferieren

Kein Docker Hub nötig -- Image wird lokal gebaut, als Tarball exportiert und direkt auf dem Login-Node zu einem SIF konvertiert:

```bash
KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
./build_and_transfer_sif.sh
```

## 2. Daten vorbereiten + hochladen

Lädt Basis-VLM + Datensatz von Hugging Face und konvertiert sie (LeRobot → HDF5 → RLDS) -- lokal, mit Internetzugang, über denselben `entrypoint.sh` wie später der Cluster-Job (nur via `docker run` statt `apptainer run`, damit die schweren Python-Abhängigkeiten nicht auf dem Host installiert sein müssen):

```bash
KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
KISSKI_PROJECT_DIR=/mnt/vast-kisski/projects/<projekt> \
HF_TOKEN=hf_... \
./prepare_and_stage_data.sh
```

## 3. Repo auf den Cluster-Projektspeicher klonen

Auf dem **Login-Node** (der hat Internet, der Compute-Node nicht):

```bash
git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git \
    /mnt/vast-kisski/projects/<projekt>/repo
```

## 4. Tokens hinterlegen + Job einreichen

```bash
printf 'hf_DEIN_TOKEN\n'  > ~/.hf_token  && chmod 600 ~/.hf_token   # nur falls SKIP_DOWNLOAD=0 gebraucht wird
printf 'DEIN_WANDB_KEY\n' > ~/.wandb_key && chmod 600 ~/.wandb_key  # optional
sbatch kisski_submit.sh
```

Volles Finetuning (Backbone nicht eingefroren) ist der Default (`FREEZE_BACKBONE=0`). Für einen Action-Head-only-Vergleichslauf mit größerem Batch als lokal:

```bash
FREEZE_BACKBONE=1 sbatch --export=ALL kisski_submit.sh
```

**Wichtig:** Env-Variablen vor `sbatch` wirken nur mit explizitem `--export=ALL` -- KISSKI setzt `SBATCH_EXPORT=none` auf dem Login-Node, das überstimmt sonst die `#SBATCH --export=ALL`-Direktive im Skript.

### Erster Testlauf (empfohlen vor einem echten 48h-Job)

```bash
MAX_STEPS=10 sbatch --export=ALL kisski_submit.sh
```

## 5. Checkpoints zurückholen

Checkpoints landen nur auf dem KISSKI-Projektspeicher -- der Compute-Node kann sie nicht selbst irgendwohin pushen (kein Internet). Abholen von dieser Maschine aus:

```bash
KISSKI_HOST=<username>@glogin-gpu.hpc.gwdg.de \
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
| [`build_and_transfer_sif.sh`](build_and_transfer_sif.sh) | Lokal bauen → Tarball → scp → `apptainer build` auf dem Login-Node. |
| [`prepare_and_stage_data.sh`](prepare_and_stage_data.sh) | Lokal Daten laden + konvertieren → rsync auf den Projektspeicher. |
| [`entrypoint.sh`](entrypoint.sh) | Container-Entrypoint (Download → Konvertierung → Training), läuft sowohl lokal (Docker) als auch im Cluster-Job (Apptainer). |
| [`kisski_submit.sh`](kisski_submit.sh) | SLURM-Batch-Skript (`sbatch kisski_submit.sh`), startet den Container mit den richtigen Binds/Env-Variablen. `SKIP_DOWNLOAD=1` per Default. |
| [`fetch_checkpoints.sh`](fetch_checkpoints.sh) | Checkpoints per rsync zurückholen, optional Push nach Hugging Face. |

## Offene Punkte / bekannte Unsicherheiten

- `entrypoint.sh` patched die Konstante `HDF5_DATA_DIR` in `rlds_dataset_g1_dex3.py` per `sed` auf den Container-Pfad (die Datei liest den Pfad nicht aus einer Env-Variable) -- funktional plausibel, aber ungetestet.
- `UnifoLM-VLM-Base`-Downloadpfad (`unitreerobotics/UnifoLM-VLM-Base`) ist nicht gegen einen echten HF-Repo-Namen verifiziert -- ggf. anpassen.
- Ressourcen-Direktiven (`-G A100:4`, `--mem=384G`, `-t 48:00:00`) sind vom GR00T-Lauf des Kollegen übernommen und müssen ggf. für UnifoLM-VLA neu kalibriert werden (anderes Modell, andere Speicherprofile).
- `KISSKI_HOST`/SSH-Zugang zum Login-Node wird vorausgesetzt, aber hier nicht eingerichtet -- eigener SSH-Key/Config nötig.
