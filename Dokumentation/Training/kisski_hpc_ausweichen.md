# Warum KISSKI HPC für das volle UnifoLM-VLA-Finetuning

## Kurzfassung

Lokal steht nur **eine** GPU zur Verfügung (NVIDIA L40S, 46 GB VRAM). Damit war bereits das Action-Head-Finetuning (VLM-Backbone eingefroren) nur mit DeepSpeed ZeRO-2 und einer sehr kleinen Batchgröße (`per_device_batch_size=2`) durchführbar -- siehe [`unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md`](unifolm_vla_scratch_finetuning_g1_dex3_blockstacking.md), Abschnitt 5 (GPU-Speicher-Probleme). Für ein **volles** Finetuning, bei dem das Qwen2.5-VL-Backbone selbst mittrainiert wird, reicht das nicht mehr aus.

## Warum genau reicht eine GPU nicht mehr aus?

Beim Action-Head-only-Training (`--trainer.freeze_modules qwen_vl_interface`) hält der Optimizer (AdamW) nur für den ungefrorenen Anteil (Action-Head + restliches Framework) zusätzliche States (Momentum + Varianz, je fp32) im Speicher -- das VLM-Backbone selbst braucht nur Platz für die (eingefrorenen) Gewichte, keine Gradienten, keine Optimizer-States.

Wird das Backbone mittrainiert, kommen pro Backbone-Parameter zusätzlich Gradienten + Optimizer-States hinzu -- bei einem Vision-Language-Modell in der Größenordnung von Qwen2.5-VL ist das ein Vielfaches des reinen Gewichts-Speicherbedarfs. DeepSpeed ZeRO-2 shardet diese States über die verfügbaren GPUs; mit nur einer GPU gibt es nichts zu shardieren -- der volle Speicherbedarf muss auf einer einzigen Karte Platz finden, was bei 46 GB VRAM (abzüglich Aktivierungen, KV-Caches, Bildern im Batch) nicht funktioniert.

## Warum KISSKI die Lösung ist

- **4× A100** (je 40 oder 80 GB, je nach zugewiesenem Knoten) statt einer L40S -- ZeRO-2 kann Optimizer-States/Gradienten tatsächlich über 4 Karten sharden.
- Deutlich mehr aggregierter VRAM erlaubt zusätzlich eine größere effektive Batchgröße (`GLOBAL_BATCH_SIZE`), was die Trainingsstabilität beim Mittrainieren des Backbones verbessert (siehe Lernraten-Anpassung in [`kisski_submit.sh`](../../Training/Kisski_Submit/kisski_submit.sh)).
- Der Kollege, dessen Repo als Vorlage diente, hat mit genau diesem Setup (SLURM + Apptainer + 4×A100) bereits erfolgreich GR00T-Finetunings auf KISSKI durchgeführt -- die Cluster-Mechanik (Token-Handling, `SBATCH_EXPORT=none`-Workaround, Container-Deployment) ist damit bereits erprobt, auch wenn das Modell selbst (UnifoLM-VLA statt GR00T) neu ist.

## Was sich dadurch ändert

| | Lokal (bisher) | KISSKI (neu) |
|---|---|---|
| GPUs | 1× L40S (46 GB) | 4× A100 |
| Backbone | eingefroren (`freeze_modules qwen_vl_interface`) | **mittrainiert** |
| `per_device_batch_size` | 2 | konfigurierbar, per Default `GLOBAL_BATCH_SIZE / NUM_GPUS` |
| Deployment | direkt via `accelerate launch` im Container/venv | Apptainer/SIF-Image, SLURM-Batch-Job (siehe [`Training/Kisski_Submit/`](../../Training/Kisski_Submit/)) |
| Ergebnis-Kennzeichnung | [`eval_videos_unifolm_vla_action_head_only/`](eval_videos_unifolm_vla_action_head_only/) | noch ausstehend |

Details zum Deployment: [`Training/Kisski_Submit/README.md`](../../Training/Kisski_Submit/README.md).
