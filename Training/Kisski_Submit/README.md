# KISSKI Submit -- UnifoLM-VLA volles Finetuning

Deployt dieses Repo per Apptainer/SIF-Container auf der KISSKI-HPC und startet das UnifoLM-VLA-Training automatisiert (Download → Konvertierung → Training, per `SKIP_*`-Flags einzeln überspringbar).

**Status: noch nicht auf echter KISSKI-Hardware getestet.** Mechanik (SLURM-Direktiven, Token-Dateien, Apptainer-Bind-Muster) ist aus dem bereits produktiv laufenden GR00T-Workflow eines Kollegen übernommen (siehe Kommentar-Header in [`kisski_submit.sh`](kisski_submit.sh)), aber auf unser eigenes Image + UnifoLM-VLA-Trainingskommando umgeschrieben -- vor dem ersten echten Lauf unbedingt mit kleinen `MAX_STEPS`/`SKIP_*`-Werten verifizieren.

## Dateien

| Datei | Zweck |
|---|---|
| [`../UnifoLM-VLA/Dockerfile`](../UnifoLM-VLA/Dockerfile) | Baut das Image (Python-Env + UnifoLM-VLA installiert). Skripte werden NICHT eingebacken, sondern zur Laufzeit gemountet. |
| [`entrypoint.sh`](entrypoint.sh) | Container-Entrypoint: Download (Basis-VLM + Datensatz) → Konvertierung (LeRobot → HDF5 → RLDS) → Training. |
| [`kisski_submit.sh`](kisski_submit.sh) | SLURM-Batch-Skript (`sbatch kisski_submit.sh`), startet den Container mit den richtigen Binds/Env-Variablen. |

## Einmaliges Setup

### 1. Image bauen + pushen (auf einer Maschine mit Docker, nicht auf KISSKI)

```bash
cd Training/UnifoLM-VLA
docker build -t <dein-dockerhub-namespace>/unifolm-vla-kisski:latest .
docker push <dein-dockerhub-namespace>/unifolm-vla-kisski:latest
```

### 2. SIF-Image auf dem Cluster erzeugen (Login-Node)

```bash
module load apptainer
mkdir -p $HOME/images
apptainer pull $HOME/images/unifolm-vla-kisski.sif \
    docker://<dein-dockerhub-namespace>/unifolm-vla-kisski:latest
```

### 3. Repo auf den Cluster-Projektspeicher klonen

Compute-Nodes haben keinen Internetzugang -- das Repo muss vorher auf dem Login-Node liegen:

```bash
git clone --depth 1 https://github.com/Fichtl00/Masterprojekt_G1_CubeStack.git \
    /mnt/vast-kisski/projects/<projekt>/repo
```

### 4. Tokens hinterlegen

```bash
printf 'hf_DEIN_TOKEN\n'  > ~/.hf_token  && chmod 600 ~/.hf_token
printf 'DEIN_WANDB_KEY\n' > ~/.wandb_key && chmod 600 ~/.wandb_key   # optional
```

## Einreichen

```bash
sbatch kisski_submit.sh
```

Volles Finetuning (Backbone nicht eingefroren) ist der Default (`FREEZE_BACKBONE=0`). Für einen Action-Head-only-Vergleichslauf mit größerem Batch als lokal:

```bash
FREEZE_BACKBONE=1 sbatch --export=ALL kisski_submit.sh
```

**Wichtig:** Env-Variablen vor `sbatch` wirken nur mit explizitem `--export=ALL` -- KISSKI setzt `SBATCH_EXPORT=none` auf dem Login-Node, das überstimmt sonst die `#SBATCH --export=ALL`-Direktive im Skript.

## Erster Testlauf (empfohlen vor einem echten 48h-Job)

```bash
MAX_STEPS=10 SKIP_DOWNLOAD=0 SKIP_CONVERT=0 sbatch --export=ALL kisski_submit.sh
```

Prüft Download, Konvertierung und die ersten Trainingsschritte, ohne die volle Walltime zu blockieren.

## Offene Punkte / bekannte Unsicherheiten

- `entrypoint.sh` patched die Konstante `HDF5_DATA_DIR` in `rlds_dataset_g1_dex3.py` per `sed` auf den Container-Pfad (die Datei liest den Pfad nicht aus einer Env-Variable) -- funktional plausibel, aber ungetestet.
- `tfds build`-Aufruf und `UnifoLM-VLM-Base`-Downloadpfad (`unitreerobotics/UnifoLM-VLM-Base`) sind nicht gegen einen echten HF-Repo-Namen verifiziert -- ggf. anpassen.
- Ressourcen-Direktiven (`-G A100:4`, `--mem=384G`, `-t 48:00:00`) sind vom GR00T-Lauf des Kollegen übernommen und müssen ggf. für UnifoLM-VLA neu kalibriert werden (anderes Modell, andere Speicherprofile).
