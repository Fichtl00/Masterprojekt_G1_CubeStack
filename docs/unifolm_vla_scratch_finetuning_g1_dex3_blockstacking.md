# UnifoLM-VLA Finetuning von Scratch auf G1_Dex3_BlockStacking_Dataset

Session-Log zum Vorhaben: UnifoLM-VLA **von Scratch** (kein G1-vortrainierter Checkpoint wie
`UnifoLM-VLA-Base`/`-Libero`) auf
[`unitreerobotics/G1_Dex3_BlockStacking_Dataset`](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset)
finetunen, mit TensorBoard-Monitoring analog zu `gr00t_finetuning_anleitung.md`. Stand:
**Training abgeschlossen** (Checkpoint `steps_24000` als final akzeptiert, siehe Abschnitt 6).

---

## 0. Ausgangslage

- Repo: `/home/omniverse-2/unifolm-vla/unifolm-vla` (NVIDIA/Qwen2.5-VL-basiertes
  UnifoLM-VLA-0, conda-env `unifolm-vla`, CUDA 12.4).
- Ziel: **von Scratch** heißt hier konkret — als VLM-Backbone dient
  `unitreerobotics/UnifoLM-VLM-Base` (reines Vision-Language-Modell, kein
  G1-Manipulations-Finetuning), **nicht** `UnifoLM-VLA-Base`/`-Libero` (das sind bereits
  auf G1-Aufgaben feingetunte VLA-Checkpoints).
- Datensatz-Ziel: `unitreerobotics/G1_Dex3_BlockStacking_Dataset` — 301 Episoden, 281196
  Frames, fps 30, **LeRobot-Codebase-Version `v3.0`**.
  - `observation.state` / `action`: flach, 28-dim, benannte Dex3-Gelenke (14 Arm + 14
    Hand pro Roboterhälfte je 7, keine Taille im State/Action enthalten).
  - 4 Kamera-Streams: `observation.images.cam_left_high`, `cam_right_high`,
    `cam_left_wrist`, `cam_right_wrist` (je 480x640, AV1-codierte Videos).
- GPU: NVIDIA L40S, ausreichend frei.

---


## 2. Klärung: 4-Kamera-Frage

**Befund:** Das Modell selbst (`QWen2_5.py::_QWen_VL_Interface`) ist kameraanzahl-agnostisch
— pro Sample wird generisch eine Liste beliebig vieler Bilder in den Chat-Prompt gepackt.
Das eigentliche Limit liegt in `datasets.py::RLDSBatchTransform.__call__`, die aktuell
**hart nur `image_primary`** verwendet, obwohl `load_camera_views` in `RLDSDataset.__init__`
mehr Kameras laden könnte, und der `use_wrist_image`-Konstruktor-Parameter existiert, aber
nirgends ausgewertet wird — ein Bug/eine Lücke im aktuellen Code-Stand.

**Abgleich mit Unitrees eigenem G1-Pretraining** (`UnifoLM-VLA-Base/config.yaml`):
`data_mix: Unitree_all_task` + `use_wrist_image: true` mappt in `datasets.py` auf
`load_camera_views = ("primary", "left_wrist", "right_wrist")` — **kein `secondary`**. In
`configs.py` ist für **alle 13** offiziellen G1-Datensatz-Einträge `"secondary": None`
hart codiert. Unitree hat selbst **nie mehr als 3 Kameras** (1 Kopf + 2 Handgelenk)
verwendet, obwohl das Schema einen vierten Slot (`secondary`) vorsieht.

**Entscheidung (Nutzer):** Bei **3 Kameras bleiben**, wie bei Unitree — `cam_left_high`
als `primary`, `cam_left_wrist`/`cam_right_wrist` als `left_wrist`/`right_wrist`.
`cam_right_high` (die zweite Kopfkamera) wird **nicht** genutzt. Kein Patch an
`RLDSBatchTransform`/`configs.py`/`datasets.py` nötig — bewährter Pfad statt unerprobtem
4-Kamera-Setup.

---

## 3. Datenformat-Kompatibilität (LeRobot v3.0) — gelöst

`LeRobotDataset` (lerobot 0.3.4 im `unifolm-vla`-Env) lädt den v3.0-Datensatz zwar korrekt
(getestet, funktioniert), materialisiert intern aber pro Episode eine **v2.1-kompatible
Legacy-Ansicht** (einzelne `episode_NNNNNN.parquet`/`.mp4`-Dateien statt der gechunkten
v3-Multi-Episode-Dateien) — dafür existiert auf dem Hub ein separater Git-Tag `v2.1` pro
Datensatz-Repo, den `LeRobotDataset` standardmäßig anfragt (`CODEBASE_VERSION = "v2.1"` ist
im installierten Paket hart codiert, unabhängig vom `codebase_version`-Feld im
`meta/info.json` des Datensatzes selbst).

**Wichtige Falle dabei:** Jede `LeRobotDataset(...)`-Instanziierung ruft — sofern nicht
lokal bereits alle für den angefragten Episoden-Slice nötigen Dateien vorhanden sind —
`get_safe_version()` → `list_repo_refs()` auf, einen Hub-API-Call, unabhängig davon ob die
Daten selbst schon lokal liegen. Mit 8-16 parallelen Worker-Prozessen (ein
`LeRobotDataset(episodes=[i])` pro Episode) hat das binnen Sekunden das HF-Rate-Limit
(429, 1000 Requests/5min) gerissen — mehrfach, auch nach vollständigem lokalem
Snapshot-Download. Zwei Ursachen kombiniert:
1. Der erste Full-Repo-Download (`hf download ... --local-dir ...`, ohne `--revision`) holt
   die **v3.0-Hauptbranch** (gechunkte `file-000.parquet`/`file-000.mp4`) — das ist aber
   NICHT das Layout, das `LeRobotDataset` tatsächlich lesen will (das ist die separate
   `v2.1`-Tag-Revision mit `episode_NNNNNN.*`-Dateien). Ergebnis: 8.6GB nutzlos geladen.
2. Selbst nach korrektem `--revision v2.1`-Download blieb der Hub-Version-Check
   (`get_safe_version`) bei jeder Instanziierung aktiv, weil `revision="v2.1"` von
   `packaging.version.parse()` als gültige Versionsnummer erkannt wird und damit *immer*
   den Hub-Refs-Abgleich auslöst (`lerobot/datasets/lerobot_dataset.py` Zeile ~482) — auch
   mit `HF_HUB_OFFLINE=1` (das bricht den Call dann nur hart ab, statt ihn zu vermeiden).

**Lösung:** Die `v2.1`-Tag-Revision einmal explizit vollständig herunterladen
(`hf download ... --revision v2.1`, mit Retry-Loop wegen Rate-Limit — zwei Versuche nötig,
der erste ließ `cam_right_wrist` und Teile von `cam_right_high` fehlend zurück, obwohl der
Befehl Erfolg meldete). Danach die zugehörige **Commit-SHA** des `v2.1`-Tags auflösen
(`HfApi().list_repo_refs(...)`) und diese SHA (nicht den String `"v2.1"`) als `revision=`
an jede `LeRobotDataset(...)`-Instanz übergeben — ein Commit-Hash ist kein gültiges
PEP440-Versions-Literal, `is_valid_version()`/`get_safe_version()` überspringen den
Hub-Call dadurch vollständig. Damit läuft die Konvertierung rein lokal, ohne weitere
Rate-Limit-Risiken.

**Für künftige Fälle geprüft (Repo-weite Suche über unifolm-vla/Groot/Unitree_VLA_stack):**
kein fertiger Konverter für dieses 28-dim-Schema vorhanden — eigene Skripte waren nötig.
Ein potenziell nützlicher Baustein existiert aber: `Groot/Isaac-GR00T/scripts/lerobot_conversion/convert_v3_to_v2.py`
(NVIDIA, getestet mit lerobot 0.4.0) konvertiert v3.0-Datensätze **lokal/offline** nach
v2.1, statt sich auf `LeRobotDataset`s lazy Hub-Materialisierung pro Episode zu verlassen
— hätte den oben beschriebenen Rate-Limit-Ärger evtl. von vornherein vermieden. Löst aber
nicht das Schema-Problem (flaches 28-dim state/action vs. Arm/Gripper/Body-Split), dafür
bleibt ein eigener HDF5-Konverter so oder so nötig.

Konverter-Skript: `prepare_data/convert_lerobot_to_hdf5_g1_dex3.py` (parallelisiert über
`multiprocessing.Pool`, ein `LeRobotDataset(episodes=[i], revision=<sha>)` pro Episode/
Worker — Video-Decoding via `pyav` ist der Flaschenhals, nicht die Modell-/State-Extraktion).
Bei 16 parallelen Workern ca. **7h Gesamtdauer** für alle 301 Episoden (I/O-/Decoder-
Kontention senkt den Pro-Worker-Durchsatz von ~10 auf ~0.7 Frames/s; CPU ist mit 32 Kernen
nicht der Engpass). Nutzerentscheidung: trotzdem laufen lassen statt Worker-Zahl zu
reduzieren.

---

## 4. Neue Datensatz-Registrierung im UnifoLM-VLA-Code

Neuer OXE-Datensatz-Key `g1_dex3_blockstacking` (28-dim Joint-Space, kein EE-Pose-Bezug),
analog zum bestehenden Muster (`g1_stack_block`), aber mit eigenem Encoding, da kein
vorhandenes `StateEncoding`/`ActionEncoding`-Enum-Mitglied zu 28 Dimensionen passt:

| Datei | Änderung |
|---|---|
| `src/unifolm_vla/rlds_dataloader/datasets/rlds/oxe/configs.py` | Neuer Enum-Wert `JOINT_G1_DEX3 = 7` (State + Action) + neuer `OXE_DATASET_CONFIGS`-Eintrag (3 Kameras: primary=cam_left_high, left_wrist, right_wrist) |
| `.../oxe/materialize.py` | `[True]*28`-Maske (`absolute_action_mask`/`action_normalization_mask`) für `JOINT_G1_DEX3` |
| `.../oxe/transforms.py` | `g1_dex3_blockstacking` → bereits vorhandene `unitree_g1_joint_dataset_transform` (reiner State/Action-Passthrough, kein EE-Konvertierungscode nötig) |
| `.../oxe/mixtures.py` | Single-Dataset-Mixture-Eintrag |
| `src/unifolm_vla/rlds_dataloader/datasets/datasets.py` | `load_camera_views = ("primary", "left_wrist", "right_wrist")` für diesen `data_mix` |
| `src/unifolm_vla/rlds_dataloader/constants.py` | `G1_DEX3_JOINT_CONSTANTS` (`ACTION_DIM=PROPRIO_DIM=28`, `NUM_ACTIONS_CHUNK=16`) + Plattform-Erkennung über `"dex3"` im Kommandozeilen-String |
| `prepare_data/hdf5_to_rlds/rlds_dataset_g1_dex3/rlds_dataset.py` (neu) | Eigener schlanker tfds-`DatasetBuilder` (3 Kameras, `state`/`action` als flache 28-dim-Tensoren, keine `ee_state`/`ee_action`-Felder wie im g1_stack_block-Pendant) |
| `prepare_data/convert_lerobot_to_hdf5_g1_dex3.py` (neu) | LeRobot(v2.1-Tag)→HDF5-Konverter für dieses Schema |
| `scripts/run_scripts/run_unifolm_vla_train_g1_dex3.sh` (neu) | Trainings-Launch-Kommando |

**Wichtige Falle bei den Modell-Dimensionen:** `src/unifolm_vla/config/training/unifolm_vla_train.yaml`
setzt `framework.action_model.action_dim`/`state_dim` standardmäßig auf `16` — das ist ein
**separater** Parameter vom RLDS-seitigen `ACTION_DIM`/`PROPRIO_DIM` in `constants.py` und
muss beim Trainingsstart per CLI-Override auf `28` gesetzt werden
(`--framework.action_model.action_dim 28 --framework.action_model.state_dim 28`), sonst
Shape-Mismatch im Diffusions-Action-Head.

TensorBoard: `train_unifolm_vla.py` nutzte bisher nur `wandb` im `offline`-Modus
(`_init_wandb`/`_log_metrics`/`_finalize_training`) — `torch.utils.tensorboard.SummaryWriter`
wurde parallel dazu ergänzt (`tensorboard` 2.15.2 war bereits über `tensorflow-datasets`
installiert, kein zusätzliches Paket nötig), Logs landen unter `{output_dir}/tensorboard`.

---

## Zusammenfassung Entscheidungen

| Frage | Entscheidung |
|---|---|
| Basismodell | `unitreerobotics/UnifoLM-VLM-Base` (reines VLM, kein G1-Finetuning) — von HF heruntergeladen |
| Kameras | 3 (wie Unitree: primary=cam_left_high + left_wrist + right_wrist), `cam_right_high` nicht genutzt |
| Speicherplatz | 4 alte Groot-Runs + doppelter HF-Cache gelöscht → 445GB frei |
| Datenformat | LeRobot v3.0 (Hub) → intern via `v2.1`-Tag-Revision (Commit-SHA gepinnt) → eigener HDF5-Konverter → eigener RLDS-Builder |
| TensorBoard | Ergänzt in `train_unifolm_vla.py`, parallel zu wandb-offline; Server läuft auf Port 6007 |
| Trainierbarer Modellumfang | VLM-Backbone (`qwen_vl_interface`) eingefroren, nur Action-Head (`action_model`, 599.6 Mio. Parameter) trainiert — volles Finetuning passt nicht auf 1x L40S |

---

## 5. Trainingsstart: GPU-Speicher & DeepSpeed-Bugs

Datenpipeline (HDF5 → RLDS, 301/301 Episoden) lief vollständig durch (~7h23min HDF5-Konvertierung,
~21min RLDS-`tfds build`). Beim eigentlichen Trainingsstart traten mehrere unabhängige Probleme auf,
bevor der Lauf stabil lief:

### 5.1 Fremde GPU-Prozesse blockierten Speicher

Ein verwaister GR00T-Policy-Server (`run_gr00t_server.py`, 4+ Tage untätig, 6.7GB VRAM) sowie zwei
Docker-Container (`isaac-lab-base`, `docker-cloudxr-runtime-1`, je 3 Tage laufend) belegten
zusammen mehrere GB VRAM. Container gestoppt (`docker stop`), GR00T-Server vom Nutzer selbst
beendet (`kill`) — GPU danach fast komplett frei (45.45GB von 46GB).

### 5.2 Volles Finetuning passt strukturell nicht auf 1x L40S (46GB)

Ein wiederkehrender `CUDA out of memory`-Fehler ("Tried to allocate 15.45 GiB") trat **identisch**
bei Batch-Size 8 und 2 auf — kein aktivierungsabhängiger (batch-skalierender) Bedarf, sondern ein
fixer Block, der exakt zu einem einzelnen vollen fp32-Gradientenpuffer für ~3.75 Mrd. Parameter
passt (3.75B × 4 Byte ≈ 15GB). Ursache: `ZeRO Stage 2` schart Optimizer-States/Gradienten nur über
**mehrere** GPUs — mit `num_processes=1` gibt es nichts zum Aufteilen, ZeRO-2 bringt also keinerlei
Speichervorteil. Rechnerisch braucht volles Finetuning (bf16-Gewichte + fp32-Master + fp32-Gradienten
+ fp32-Adam-Momente) für das 8.9-Mrd.-Parameter-Modell (Qwen2.5-VL-Backbone + Action-Head) ca.
65-70GB — mehr als jede einzelne verfügbare Karte.

**Recherche bestätigt das als bekanntes Muster:** OpenVLA-artige VLA-Modelle werden für volles
Finetuning typischerweise auf **8x A100** trainiert; ein verwandtes Unitree-Repo
(`unifolm-world-model-action`, Issue #35) hatte selbst auf 4x24GB (96GB) mit ZeRO-3 + CPU-Offload +
Batch-Size 1 ungelöste `CUDA out of memory`-Probleme. Kein LoRA/PEFT-Support im Repo vorhanden
(`peft` nicht installiert, Code-Treffer für "lora" waren nur Substring-Zufälle wie
"exploration"-Datensatznamen).

**Entscheidung (Nutzer):** VLM-Backbone einfrieren (`--trainer.freeze_modules qwen_vl_interface`),
nur der Diffusions-Action-Head (`action_model`) wird trainiert. Reduziert trainierbare Parameter
von 8.9 Mrd. auf **599.6 Mio.** — Optimizer-Speicherbedarf dadurch unproblematisch für 1 GPU.
Fixe Modelldateien: `self.qwen_vl_interface` (Qwen2.5-VL-Backbone) und `self.action_model`
(Flow-Matching-Action-Head) sind die zwei Top-Level-Submodule in
`src/unifolm_vla/model/framework/unifolm_vla.py`.

### 5.3 Zwei Code-Bugs beim Zusammenspiel Freeze + DeepSpeed (gepatcht)

1. **`build_param_lr_groups()`** (`src/unifolm_vla/training/trainer_utils/metrics.py`) filterte
   Parameter-Gruppen nicht nach `requires_grad` — gepatcht, um eingefrorene Parameter
   auszuschließen (leere Gruppen werden komplett übersprungen statt an den Optimizer übergeben).
2. **Eigentliche Root Cause:** Der Optimizer wird in `main()` **vor** dem Freeze aufgebaut
   (`setup_optimizer_and_scheduler()` läuft vor `VLATrainer.prepare_training()`, wo das Freezing
   passiert). DeepSpeeds `BF16_Optimizer._setup_for_real_optimizer()` filtert Parameter-Gruppen
   selbst nochmal nach `requires_grad`
   (`deepspeed/runtime/bf16_optimizer.py`) — eine zum Zeitpunkt des Optimizer-Baus noch
   ungefrorene, komplett dem VLM zugehörige Gruppe wird dadurch zur Trainingszeit leer und lässt
   `torch._C._nn.flatten_dense_tensors` mit `torch.cat(): expected a non-empty list of Tensors`
   abstürzen. **Fix:** `TrainerUtils.freeze_backbones(...)` wird jetzt in `main()` direkt nach
   `build_framework(cfg)` aufgerufen — **vor** `setup_optimizer_and_scheduler()` — statt (nur)
   in `VLATrainer.prepare_training()`.
3. Zusätzlich `ZeRO Stage` in `ds_config.yaml` von 2 auf **0** gesetzt (kein Sharding nötig bei
   1 GPU, vermeidet weitere ZeRO-2-spezifische Edge Cases).

### 5.4 TensorBoard-Paket brauchte ein Downgrade

`tensorboard`-CLI/-Modul scheiterte mit `ModuleNotFoundError: No module named 'pkg_resources'` —
neuere `setuptools`-Versionen (≥81) haben `pkg_resources` entfernt. Fix: `pip install "setuptools<81"`.

### 5.5 Ergebnis: Training läuft stabil

Nach allen Fixes: ~3.6 Schritte/Sekunde, GPU-Auslastung 34GB/46GB (komfortabel), Loss bei Schritt
100 ≈ 1.07 (plausibler Startwert für Flow-Matching-Action-Loss). Bei dieser Rate: 30.000 Schritte
in ca. **2.3 Stunden** (viel schneller als zunächst befürchtet, da nur der kleine Action-Head
trainiert wird).

- **TensorBoard-Server:** `http://<host-ip>:6007` (Port 6007 statt 6006, da dort noch ein alter
  GR00T-TensorBoard-Prozess lief), Logdir `/home/omniverse-2/Groot/outputs_unifolm_vla`.
- **Output-Verzeichnis:** `/home/omniverse-2/Groot/outputs_unifolm_vla/g1_dex3_blockstacking_scratch/`
- **Run-Script:** `scripts/run_scripts/run_unifolm_vla_train_g1_dex3.sh`

---

## 6. Trainingsabbruch bei Schritt ~26000: VM-Disk voll (gelöst)

Nach dem stabilen Start (Abschnitt 5.5) lief das Training bis kurz nach Schritt 24000 weiter,
brach dann aber ab: die **VM-Root-Disk** lief voll (nicht der GPU/System-RAM). Ursache: Jeder
Checkpoint speichert die vollen Modellgewichte unkomprimiert (`torch.save(state_dict, ...)`,
~17.8GB pro Checkpoint) bei `--trainer.save_interval 2000` — nach 12 Checkpoints (Schritt 2000
bis 24000) waren das bereits ~213GB, und die Disk war zu diesem Zeitpunkt ohnehin schon bei
99% (nur 25GB frei, siehe Abschnitt 1). Der letzte Checkpoint (`steps_26000`, nur 6.8GB statt
17.8GB) wurde mitten im Schreiben abgeschnitten und ist unbrauchbar — gelöscht.

**Vorhandene, vollständige Checkpoints:** `steps_2000` bis `steps_24000` (Schritte 2000, 4000,
6000, ..., 24000 — je ~17.8GB), alle intakt.

### 6.1 Root-Disk erweitert

Die VM-Disk wurde vom Nutzer auf Hypervisor-Ebene von 2TB auf 3.9TB vergrößert; die
Partition/LVM/das Dateisystem mussten das aber erst noch übernehmen (`sda3` blieb bei 2TB,
obwohl `sda` schon 3.9TB zeigte). Online-Vergrößerung (kein Datenverlust-Risiko, da nur
Wachstum, kein Shrink) durchgeführt:

```bash
sudo growpart /dev/sda 3                                          # Partition erweitern
sudo pvresize /dev/sda3                                           # LVM Physical Volume erweitern
sudo lvextend -l +100%FREE /dev/mapper/ubuntu--vg-ubuntu--lv       # Logical Volume erweitern
sudo resize2fs /dev/mapper/ubuntu--vg-ubuntu--lv                   # ext4-Dateisystem erweitern
```

Ergebnis: `/` von 25GB auf **1.9TB frei** (bei 3.9TB Gesamtgröße, 50% belegt).

### 6.2 Kein sauberes Resume möglich — Bug im Checkpointing

`_save_checkpoint()` (`train_unifolm_vla.py`) speichert nur die rohen Modellgewichte
(`torch.save(state_dict, ...)`), **nicht** Optimizer-Zustand, LR-Scheduler oder
Schritt-Zähler. Die vorhandene `is_resume`-Logik (`_load_checkpoint` →
`self.accelerator.load_state(checkpoint_path)`) erwartet aber ein vollständiges
Accelerate-`save_state()`-Verzeichnis (mit `optimizer.bin`, `scheduler.bin`,
`random_states.pkl` etc.) — das wird nie geschrieben. Ein echtes 1:1-Fortsetzen bei Schritt
24000 (mit fortgesetztem Optimizer-Momentum und LR-Schedule) ist mit diesem Code-Stand also
**nicht möglich**, nur ein Neuladen der Gewichte als Initialisierung mit bei Schritt 0
neu startendem Optimizer/LR-Schedule.

**Entscheidung (Nutzer):** `steps_24000_pytorch_model.pt` (80% des geplanten Trainings, Loss
war bereits deutlich gesunken) als **finalen Checkpoint** akzeptieren, kein Weitertrainieren.

### 6.3 Ergebnis

- **Finaler Checkpoint:** `/home/omniverse-2/Groot/outputs_unifolm_vla/g1_dex3_blockstacking_scratch/checkpoints/steps_24000_pytorch_model.pt`
- **Alle Zwischencheckpoints** (2000–22000) sind ebenfalls vorhanden, falls ein früherer Stand
  zum Vergleich (z.B. Overfitting-Check) gebraucht wird.
- **TensorBoard** (Trainingskurven bis Schritt ~26000, letzter unvollständiger Log-Eintrag
  ignorierbar) weiterhin unter Port 6007 abrufbar.
- **Offener Punkt für künftige Trainingsläufe mit diesem Repo:** Falls längeres Training mit
  Resume-Bedarf ansteht, müsste `_save_checkpoint()`/`_load_checkpoint()` auf
  `self.accelerator.save_state(...)`/`load_state(...)` umgestellt werden (echter Trainer-State
  statt nur Modellgewichte), und/oder `save_interval` großzügiger gewählt bzw. alte
  Checkpoints automatisch rotiert werden, um Disk-Verbrauch zu begrenzen.

---

## 7. Open-Loop-Evaluation

Kein fertiges Open-Loop-Eval-Skript in UnifoLM-VLA vorhanden (nur LIBERO-Sim- und
Real-Robot-Server-Eval unter `scripts/eval_scripts/`) — analog zu GR00Ts
`gr00t/eval/open_loop_eval.py` (siehe `referenz_agile_joint_space_finetuning_und_eval.md`) neu
geschrieben: `scripts/tools/open_loop_eval_g1_dex3.py`.

**Vorgehen:** Für 5 Trajektorien (Episoden 0-4, **Teil der 301 Trainings-Episoden, kein
Hold-out** — gleicher Vorbehalt wie beim Referenz-Vorbild) wird an jedem `stride`-ten Zeitschritt
mit den *aufgezeichneten* Kamerabildern + Zustand ein Action-Chunk (16 Schritte) vorhergesagt
und mit der aufgezeichneten Ground-Truth-Aktion verglichen (MSE/MAE), keine Simulation
involviert. Nutzt `baseframework.from_pretrained(ckpt_path, vlm_pretrained_path=...)` +
`vla.predict_action(qwen_inputs=...)` — Inferenz-API wie in
`deployment/model_server/run_real_eval_server.py`, aber direkt gegen die schon konvertierten
HDF5-Episoden statt live über die Deployment-Server-API.

**Ergebnis** (Checkpoint `steps_24000`, Stride 40):

| Episode | MSE | MAE |
|---|---|---|
| 0 | 0.0423 | 0.1260 |
| 1 | 0.0470 | 0.1299 |
| 2 | 0.0416 | 0.1384 |
| 3 | 0.0542 | 0.1501 |
| 4 | 0.0351 | 0.1187 |
| **Ø** | **0.0440** | **0.1326** |

Vergleichbar mit den GR00T/Referenz-Referenzwerten (MSE ⌀0.035, MAE ⌀0.125, siehe
`referenz_agile_joint_space_finetuning_und_eval.md` Abschnitt 3) — konsistent niedrige Fehler über
alle 5 Trajektorien, kein Ausreißer. Beschriftete Pro-Gelenk-Plots (GT vs. Vorhersage über die
Zeit, alle 28 Dex3-Gelenke einzeln benannt) unter
`.../g1_dex3_blockstacking_scratch/eval_open_loop/traj_{0..4}_open_loop.jpeg`.

**Gleicher Vorbehalt wie beim Referenz-Vorbild:** niedriger Open-Loop-Fehler zeigt gutes Fitten der
Trainingsdaten, **nicht** Generalisierung — kein Hold-out-Split vorhanden. Für eine echte
Generalisierungsaussage bräuchte es 3-5 nie trainierte Episoden zum Vergleich.

---

## 8. Closed-Loop-Evaluation

### 8.1 SHM-Bridge (`cubestack_jointspace_unifolm_vla_bridge.py`) — gescheitert, nicht weiterverfolgt

Die vorbereitete, aber nie end-to-end getestete SHM-Bridge (Env
`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace`, siehe
`cube_stack_eval_env.md`) hing bei jedem Versuch reproduzierbar direkt nach
`"Starting the simulation..."` fest, bei unkontrolliert wachsendem Host-RAM (bis zu ~90GB,
manuell abgebrochen bevor der Host erneut OOM ging). Reproduziert sowohl mit `--device cpu`
als auch `--device cuda` — identischer Hängepunkt, identisches RAM-Wachstum, also kein
Rendering-Performance-Problem, sondern ein echter Bug in der Bridge selbst (Docstring
behauptet "CONFIRMED WORKING", widerspricht aber dem in `cube_stack_eval_env.md`
dokumentierten Status "noch NICHT end-to-end getestet" — vermutlich ein nachträglich
ergänzter, nie verifizierter Kommentar). Nicht weiter debuggt.

### 8.2 Architektur-Wechsel: GR00T-Muster statt SHM

Entscheidung (Nutzer): Auf das bereits bei GR00T bewährte Muster umsteigen — **ein** Prozess
treibt Env **und** Eval-Loop synchron per Request/Response an einen Policy-Server, statt
zweier asynchron über SHM gekoppelter Prozesse (vgl. `groot_vs_unifolm_vla_vergleich.md`,
Abschnitt 1: "ZMQ liefert klare Episodengrenzen ... kostenlos", SHM hat "mehr bewegliche
Teile ... als beim ZMQ-Pfad nötig war").

Statt eines neuen ZMQ-Servers wird der bereits vorhandene, für LIBERO genutzte
FastAPI-HTTP-Server wiederverwendet:

- **Server (Host)**: `deployment/model_server/run_real_eval_server.py` — unverändert, nur mit
  unseren Parametern gestartet (`--ckpt_path .../steps_24000_pytorch_model.pt`,
  `--vlm_pretrained_path /home/omniverse-2/UnifoLM-VLM-Base`,
  `--unnorm_key g1_dex3_blockstacking`). Antwortet auf `POST /act` mit dem vollen
  16-Schritt-Action-Chunk (nicht nur einem Schritt) — passt exakt zum
  GR00T-Eval-Loop-Muster (Chunk anfordern, `action_horizon` Schritte selbst iterieren).
- **Eval-Loop (Container)**: neu geschrieben, `scripts/tools/eval_cubestack_unifolm_vla.py`
  (Vorbild: `eval_referenz_korb_ausladen_agile.py`) — `gym.make(...)`, `env.step()` und
  HTTP-Client in einem Prozess. Kein Joint-Remapping nötig (anders als bei Referenz): die
  JointSpace-Env liefert `arm_hand_joint_pos_ordered` (28-dim) und erwartet die Aktion
  bereits exakt in Datensatz-Reihenfolge.
- **Stolperstein 1**: `--enable_pinocchio`-Flag existierte nur in der Bridge, nicht in einem
  neuen Skript für dieselbe Env — ohne vorherigen `import pinocchio` vor dem Import von
  `isaaclab_tasks.manager_based.locomanipulation.pick_place` schlägt der Package-Import fehl
  (das Package importiert beim Laden die Pink-IK-Sibling-Envs mit). Übernommen.
- **Stolperstein 2**: Container läuft im Docker-Netzwerk-Modus `host` — `--server_host` muss
  `127.0.0.1` sein, nicht die Docker-Bridge-Gateway-IP (`172.17.0.1`).
- **Stolperstein 3**: erster Request schlug mit `422 Unprocessable Entity` fehl — `requests.post(..., data=...)`
  setzt ohne expliziten Header kein `Content-Type: application/json`, FastAPI konnte den Body
  dadurch nicht parsen. Fix: `headers={"content-type": "application/json"}` gesetzt.
- **`json_numpy`** war im Isaac-Sim-Python-Env des Containers nicht installiert, nachinstalliert
  (`_isaac_sim/python.sh -m pip install json_numpy`).

### 8.3 Smoke-Test: Architektur funktioniert

2 Episoden, 100 max_steps, `--device cuda`: beide liefen sauber durch (112 Schritte, ~13s
je Episode), RAM stabil (~16GB, kein Wachstum), Server antwortete durchgehend mit `200 OK`
(~0.14s Inferenzzeit nach Cache-Warmup). Ergebnis (`eval_results_unifolm_vla.json`):

| Episode | Erfolg | Steps | Dauer |
|---|---|---|---|
| 1 | nein | 112 | 13.6s |
| 2 | nein | 112 | 12.7s |

0/2 Erfolge bei nur 100 Schritten (≈6 Action-Chunks) für eine 3-Würfel-Stapelaufgabe ist
für sich genommen nicht überraschend. Alle Artefakte gesichert unter
`/home/omniverse-2/IsaacLab/eval_videos_unifolm_vla/`
(`episode_001_grid.mp4`, `episode_002_grid.mp4`, `eval_results_unifolm_vla.json`,
`frames_comparison/` — extrahierte Einzelbilder).

### 8.4 Video-Review: vermutlich massive Sim-Real-Domain-Gap statt Modellfehler

Qualitative Durchsicht beider Videos (Frames über die volle Episode verteilt extrahiert):
Episode 1 zeigt zumindest zielgerichtete Bewegung des linken Arms in Richtung der Würfel
(kein Griff/Stapeln erreicht); Episode 2 wirkt über die komplette Episode nahezu regungslos
(Start- und Endpose praktisch identisch). Kein sichtbarer Griff- oder Stapelversuch in
keiner der beiden Episoden. Der rechte Arm bleibt in beiden Fällen passiv — das ist
allerdings **kein Fehlverhalten**, sondern deckt sich mit dem realen Trainingsvideo (s.u.),
wo der rechte Arm ebenfalls meist passiv bleibt (überwiegend einarmige Aufgabe).

**Entscheidender Fund beim Gegencheck mit einer echten Trainings-Episode**
(`episode_0.hdf5`, Frames über die volle Episode extrahiert,
`frames_comparison/train_ep0_*.jpg`): Die Trainingsdaten sind **reale Kamerabilder eines
echten G1-Roboters** — fotorealistisch, echte Beleuchtung/Schatten, echte Holzwürfel auf
einem echten weißen Tisch, sichtbares Robotergehäuse/Kabel. Die Isaac-Sim-Renderings in
unserer Closed-Loop-Eval sehen dagegen stilisiert/kontrastarm aus (grau, kantige
Low-Poly-Optik, Schachbrett-Boden, andere Materialien/Beleuchtung) — eine **massive visuelle
Domänen-Lücke**, keine kosmetische Nebensache.

Da der VLM-Backbone (`qwen_vl_interface`) beim Finetuning eingefroren war (Abschnitt 5.2) und
nie an synthetische Bilder angepasst wurde, sind die visuellen Repräsentationen, die er aus
den Sim-Renderings extrahiert, wahrscheinlich stark "out of distribution" gegenüber dem, was
der (trainierte) Action-Head interpretieren gelernt hat. Das ist eine plausiblere Erklärung
für das ziellose/passive Verhalten in der Sim als ein Trainings- oder Code-Fehler — und ein
strukturelles Problem der Eval-Methode, nicht des Checkpoints selbst.

### 8.5 Entscheidung: Closed-Loop-Test hier abgeschlossen

**Entscheidung (Nutzer):** Kein voller Lauf (5 Episoden/500 Steps) mehr — die
IsaacLab-Sim-Eval ist für ein ausschließlich auf echten Kamerabildern trainiertes Modell
strukturell kein aussagekräftiger Test, ein längerer Lauf würde eher die Domain-Gap als die
Modellqualität messen. Das **Open-Loop-Ergebnis** (Abschnitt 7, Fit auf echten
Trainingsdaten) bleibt die verlässlichste in dieser Session verfügbare Kennzahl für dieses
Modell. Ein aussagekräftiger Closed-Loop-Test würde entweder photorealistischeres
Sim-Rendering (Domain Randomization/bessere Materialien/Beleuchtung) oder eine Evaluation auf
echter Hardware erfordern — beides außerhalb des Umfangs dieser Session.

