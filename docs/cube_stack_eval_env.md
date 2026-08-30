# Zweite Eval-Umgebung: Cube Stacking (GR00T / UniVLA Vergleich)

## Zweck

Neben der Referenz-spezifischen Umgebung (`Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-Referenz-Korb-ausladen`)
gibt es jetzt eine zweite, task-neutrale Eval-Umgebung fuer den Vergleich von
**GR00T N1.7** und **UniVLA** am Standard-Benchmark "Cube Stacking" (3 Wuerfel:
rot / gelb / gruen), statt am Referenz-Korb-Task. Grund: Cube Stacking ist ein
weit verbreiteter VLA-Benchmark-Task (siehe z.B. `unitree_sim_isaaclab/tasks/g1_tasks/
stack_rgyblock_g1_29dof_dex3/`), wodurch sich beide Policies leichter mit
Literatur-/Community-Ergebnissen vergleichen lassen als am proprietaeren Referenz-Task.

## Neue Datei

`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py`

Registriert in `.../pick_place/__init__.py` als Gym-Task-ID:

```
Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack
```

**Status:** per Teleoperation (Dex3-Variante) bereits erfolgreich getestet -- Szene,
Kameras und Greifer-Kontaktphysik funktionieren.

**Update (Cube-Spawn-Positionen 10 cm naeher am Roboter):** Nach dem ersten Teleop-Test
wurden die Wuerfel-Startpositionen (`CUBE_RED/YELLOW/GREEN_INIT_POS`) in **beiden**
Varianten (Dex3 hier und Dex1/ParallelGripper) um 10 cm entlang der Vorwaerts-Achse (Y,
Richtung Roboter) verschoben: `y = 0.33/0.45/0.57` -> `y = 0.23/0.35/0.47` (X/Z sowie der
12-cm-Abstand zueinander unveraendert). Abgleich gegen die Roboter-Wuerfel-Distanz im
unitree_sim_isaaclab-Referenz-Task (`base_scene_stack_rgyblock.py`: Roboter
`(-4.2,-3.7,0.76)`, Wuerfel um `(-4.1..-4.25, -4.05..-4.12, 0.84)`, also ca. 0.38-0.42 m
Vorwaerts-Abstand vom Roboter-Pelvis) zeigt, dass die neuen Positionen (ca. 0.30-0.50 m
Abstand vom G1-Pelvis bei `(0,0,0.75)`) in einer vergleichbaren Groessenordnung liegen.

Tischgroesse gegengeprueft (`packing_table.usd`, per `UsdGeom.BBoxCache`): 2.47 m x 0.76 m,
Ursprung mittig. Der naechste Wuerfel (rot, 0.32 m Y-Versatz vom Tischursprung) liegt damit
noch auf dem Tisch, aber mit nur ~3.9 cm Rand zur Tischkante -- eng, aber ok (per Teleop
bestaetigt erreichbar).

**Update (10 cm nach rechts aus Robotersicht):** Nach Bestaetigung der Erreichbarkeit
zusaetzlich alle 3 Wuerfel um 10 cm nach rechts (aus Sicht des Roboters) verschoben:
`x = -0.35` -> `x = -0.25` (Y unveraendert). Da der G1 mit `rot=(0.7071,0,0,0.7071)`
(90 Grad um Z) gespawnt wird, entspricht "rechts" in dieser Szene einer Verschiebung in
Richtung **+X** in Weltkoordinaten (lokal +X = vorne -> Welt +Y; lokal -Y = rechts -> Welt
+X).

## Design-Entscheidungen

### Beobachtungs-/Aktionsformat identisch zur Referenz-Ausladen-Umgebung

Damit das bestehende Eval-Script (`scripts/tools/eval_referenz_korb_ausladen.py`, siehe
`closed_loop_evaluation_zusammenfassung.md`) und der eingebettete GR00T-ZMQ-Client
moeglichst unveraendert (nur Task-ID + Language-Instruction anpassen) fuer den
Cube-Stack-Vergleich weiterverwendet werden koennen, wurden 1:1 uebernommen:

- **4 Kameras**: `cam_left_high`, `cam_right_high`, `cam_left_wrist`, `cam_right_wrist`
  (gleiche Offsets/Konvention wie in der Referenz-Ausladen-Env, da dieselbe G1-Hardware/
  Kamerabestueckung).
- **Action Term**: `G1_UPPER_BODY_IK_ACTION_CFG` (Pink-IK-Loeser, siehe
  `configs/pink_controller_cfg.py`) -- 28-dim rohe Policy-Ausgabe
  (`left_arm`/`right_arm` 6D-Pose + `left_hand`/`right_hand` Fingerziele).
- **State-Observations**: `robot_joint_pos`, `left_eef_pos/quat`, `right_eef_pos/quat`
  (identische Keys zum GR00T-Observation-Schema).

### Was sich gegenueber der Referenz-Ausladen-Umgebung geaendert hat

- **Objekte**: statt Montagekorb + Inlay jetzt 3 Wuerfel (`cube_red`, `cube_yellow`,
  `cube_green`), 4.5 cm Kantenlaenge, als `RigidObjectCfg`/`CuboidCfg` (kein externes
  USD-Asset noetig, analog zu `unitree_sim_isaaclab/tasks/common_scene/
  base_scene_stack_rgyblock.py`).
- **Reset-Logik** (`reset_cubes_disjoint`): alle 3 Wuerfel werden bei jedem Reset
  unabhaengig voneinander innerhalb eines kleinen Fensters (±3 cm) um ihre
  Basisposition neu platziert -- die Basispositionen liegen 12 cm auseinander, sodass
  die Fenster nie ueberlappen. Der Roboter muss den Stack in jeder Episode komplett neu
  aufbauen (kein Fixed-Distance-Pairing wie im unitree-Referenz-Task).
- **Erfolgskriterium** (`cubes_stacked_trihand`): eigene, schlanke Stapel-Pruefung
  (xy-Abstand + Hoehen-Differenz zwischen rot-gelb und gelb-gruen), **ohne**
  Greifer-Offen-Check. Grund: die vorhandenen Utility-Funktionen `object_stacked`/
  `cubes_stacked` in `isaaclab_tasks.manager_based.manipulation.stack.mdp` setzen einen
  einfachen Parallelgreifer mit `gripper_open_val` voraus -- die G1-Dreifinger-Hand
  (Dex3/Trihand) hat kein solches Konzept, daher eigene Funktion ohne diesen Check.
- **Sprachinstruktion**: `TASK_DESCRIPTION` in der neuen Datei
  ("Stack the red cube on the yellow cube, then stack the green cube on top.") --
  muss beim Eval-Script als `annotation.human.action.task_description` (identischer
  Language-Key wie beim Referenz-Task, siehe `closed_loop_evaluation_zusammenfassung.md`)
  uebergeben werden.

### Bewusst NICHT uebernommen

- Kontaktsensor-basierte Subtask-Signale (`reach_left/right`, `idle_left/right`) und
  die zugehoerige Isaac-Lab-Mimic-Konfiguration der Referenz-Ausladen-Env -- fuer eine reine
  Closed-Loop-Policy-Evaluation (kein Mimic-Annotieren) nicht noetig. Falls spaeter auch
  fuer Cube Stacking Demonstrationsdaten per Mimic annotiert werden sollen, muesste
  analog zu `referenz_korb_ausladen_g1_mimic_env_cfg.py` eine eigene Mimic-Env-Klasse mit
  Subtask-Kontaktsensoren ergaenzt werden.

## Zweite Variante: Parallelgreifer (UniVLA-Test)

### Bereits vorhanden: unitree_sim_isaaclab

Fuer einen Test des **vortrainierten UniVLA** (Parallelgreifer statt Dreifinger-Dex3-
Hand) existiert bereits `unitree_sim_isaaclab/tasks/g1_tasks/
stack_rgyblock_g1_29dof_dex1/`, direkt neben `stack_rgyblock_g1_29dof_dex3/`, registriert
als Gym-Task `Isaac-Stack-RgyBlock-G129-Dex1-Joint` (Env-Klasse
`StackRgyBlockG129DEX1BaseFixEnvCfg`). Aufbau: gleiche geteilte Szene
`tasks.common_scene.base_scene_stack_rgyblock.TableRedGreenYellowBlockSceneCfg`, nur mit
`G1RobotPresets.g1_29dof_dex1_base_fix` (2-Gelenk-Parallelgreifer) und
`CameraPresets.left/right_gripper_wrist_camera()`. Beobachtungen: `robot_joint_state` +
`robot_gipper_state` (2 Gelenke) statt `robot_dex3_state` (14 Gelenke).

### Neu portiert: IsaacLab (analog zur GR00T-CubeStack-Env oben)

Damit auch in **IsaacLab** (statt nur in unitree_sim_isaaclab) ein Parallelgreifer-Env
fuer den UniVLA-Test zur Verfuegung steht -- mit demselben 4-Kamera-Beobachtungsformat wie
die Dex3-CubeStack-Env --, wurde `stack_rgyblock_g1_29dof_dex1` nach IsaacLab portiert:

**Neue Datei:**
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg_CubeStack_ParallelGripper.py`

**Gym-Task-ID:** `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-ParallelGripper`

**Was beim Portieren angepasst wurde:**

- **Roboter-Asset**: `G1_29DOF_DEX1_CFG` (in der neuen Datei definiert, 1:1 aus
  `unitree_sim_isaaclab/robots/unitree.py` -> `G129_CFG_WITH_DEX1_BASE_FIX` uebernommen:
  gleiche Gelenke/Actuator-Tuning). Das zugehoerige USD
  (`g1_29dof_with_dex1_base_fix1.usd`, ~77 MB inkl. `configuration/`-Unterordner) wurde
  von `unitree_sim_isaaclab/assets/robots/g1-29dof-dex1-base-fix-usd/` nach
  `source/isaaclab_assets/custom_assets/G1_Dex1_ParallelGripper_usd/` **kopiert** --
  der unitree_sim_isaaclab-Ordner ist im IsaacLab-Docker-Container nicht gemountet
  (siehe `docker/docker-compose.yaml`, kein entsprechendes Volume), ein direkter Pfad
  auf das Original haette dort also nicht aufgeloest werden koennen.
- **Action: direkte Gelenkwinkel-Steuerung statt Pink-IK** (siehe Debug-Verlauf unten fuer
  die Begruendung) -- `JointPositionActionCfg` auf den 12 Arm-/Taille-/Greifer-Gelenken
  (`UPPER_BODY_JOINT_NAMES`), analog zu `stack_rgyblock_g1_29dof_dex1_joint_env_cfg.py`,
  das fuer denselben Roboter ebenfalls keine IK verwendet. Aktionsformat damit **nicht**
  identisch zur Dex3-Env (dort: 28-dim 6D-Handgelenkpose+Hand via Pink-IK), sondern
  21-dim rohe Gelenkwinkel -- fuer die UniVLA-Auswertung entsprechend im Joint-Space statt
  im Pose-Space.
- Kein Hand-Tracking-/Motion-Controller-Teleop mehr verdrahtet -- der urspruenglich
  geplante `G1TriHandUpperBodyMotionControllerGripperRetargeterCfg` (Trigger-Greifer +
  Handgelenkpose) setzte die Pink-IK-Pose-Aktion voraus, die es jetzt nicht mehr gibt.
  Diese Env ist fuer skriptgesteuerte Policy-Evaluation (UniVLA) gedacht, nicht fuer
  Hand-Tracking-Teleoperation.
- Szene (3 Wuerfel), Reset-Logik (`reset_cubes_disjoint`) und Erfolgscheck
  (`cubes_stacked_parallel_gripper`) sind inhaltlich identisch zur Dex3-CubeStack-Env
  uebernommen (dupliziert, nicht importiert -- siehe Konvention der bestehenden
  pick_place-Envs, jede Datei ist bewusst eigenstaendig lauffaehig).

**Debug-Verlauf (2 Bugs, beide per Teleop-Testlauf gefunden und behoben):**

1. *Kamera-Pfad-Fehler.* Erster Testlauf
   (`teleop_se3_agent.py --task ...CubeStack-ParallelGripper --teleop_device handtracking
   --enable_pinocchio --headless`) scheiterte mit:
   ```
   ERROR: Failed to create environment: Unable to find source prim path:
   '/World/envs/env_.*/Robot/torso_link/d435_link'. Please create the prim before spawning.
   ```
   Ursache: Kamera-Prim-Pfade 1:1 von der Dex3-CubeStack-Env kopiert
   (`Robot/torso_link/d435_link/...`, `Robot/left_wrist_yaw_link/...`) -- die Dex1-USD hat
   aber eine andere Link-Topologie. Abgleich gegen
   `unitree_sim_isaaclab/tasks/common_config/camera_configs.py` (dort bereits korrekt fuer
   denselben Parallelgreifer konfiguriert) ergab die richtigen Mount-Punkte: Kopfkamera
   `Robot/d435_link/...` (direkt unter `Robot`), Handgelenkkameras
   `Robot/left/right_hand_base_link/...`. Beide korrigiert (Pfade + Offsets aus
   `left/right_gripper_wrist_camera`). Der stereoskopische Links/Rechts-Split der
   Kopfkamera (±0.03 m X) bleibt eine eigene Ergaenzung dieser Datei (unitree's
   `g1_front_camera` ist nur eine einzelne Kamera) -- visuell in Isaac Sim noch
   gegenpruefen.

2. *Pink-IK-Fehler (Grund fuer den Wechsel auf Gelenkwinkel-Steuerung).* Nach dem
   Kamera-Fix (`--teleop_device keyboard`) kam:
   ```
   ERROR: Failed to create environment: 'left_hand_index_0_joint' is not in list
   ```
   Ursache: `isaaclab/controllers/pink_ik/pink_ik.py::_setup_joint_ordering_mappings`
   baut eine vollstaendige Namens-Abbildung zwischen ALLEN Gelenken der Kinematik-URDF
   (`g1_29dof_with_hand_only_kinematics.urdf`) und den tatsaechlichen Gelenknamen der
   Roboter-USD auf -- nicht nur den von Pink tatsaechlich geloesten Arm-/Taille-Gelenken,
   wie urspruenglich (falsch) angenommen. Die URDF wurde fuer die Dex3-Hand generiert (14
   Fingergelenke wie `left_hand_index_0_joint`); die Dex1-USD hat nur
   `left_hand_Joint1_1`/`Joint2_1`. Eine passende Dex1-Kinematik-URDF existiert nirgends
   (IsaacLab hat nur einen URDF->USD-Konverter, keinen umgekehrten; unitree_sim_isaaclab
   hat keine `.urdf`-Quelldateien, nur USD). Da `stack_rgyblock_g1_29dof_dex1` im
   unitree-Referenz-Task ebenfalls keine IK verwendet, wurde hier genauso auf
   `JointPositionActionCfg` umgestellt (siehe oben) statt eine eigene URDF zu erzeugen.

**Noch offen / vor dem naechsten Lauf zu pruefen:**
- Der Wechsel auf `JointPositionActionCfg` ist noch NICHT per Teleop/Env-Erstellung
  gegengetestet (naechster Schritt).
- `left_wrist_yaw_link`/`right_wrist_yaw_link` (eef-Observations) sowie
  `left/right_hand_base_link` (Handgelenkkameras) werden als auf der Dex1-USD vorhanden
  angenommen (gleiche Arm-Kette wie Dex3), aber noch nicht einzeln bestaetigt.

**Status:** Datei + Asset-Kopie + Registrierung vorbereitet, 2 Bugs per Teleop-Testlauf
gefunden und behoben (Kamera-Pfade, Pink-IK -> Joint-Space-Wechsel), **Fix noch nicht
erneut gegengetestet**.

## Dritte Variante: Dex3 + Joint-Space (UniFolm-VLA)

Nutzer-Hinweis: UniFolm-VLA arbeitet auch fuer den Dex3-Roboter im Joint-Space (nicht nur
fuer den Parallelgreifer). Daher zusaetzlich zur bestehenden Pink-IK/GR00T-Dex3-Env
(`fixed_base_upper_body_ik_g1_env_cfg_CubeStack.py`, **bleibt unveraendert**) eine dritte
Datei angelegt, die dieselbe Dex3-Hand mit direkter Gelenkwinkel-Steuerung kombiniert:

**Neue Datei:**
`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/fixed_base_upper_body_ik_g1_env_cfg_CubeStack_JointSpace.py`

**Gym-Task-ID:** `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace`

**Vorabklaerung, Runde 1 (Recherche in `/home/omniverse-2/Unitree_VLA_stack/vla/unifolm-vla/`,
generischer Repo-Code):** ergab zunaechst einen scheinbaren Widerspruch -- der generische
`g1_stack_block`-Datensatzeintrag in `rlds_dataloader/datasets/rlds/oxe/configs.py` ist auf
das **`EE_R6_G1`-Pose-Encoding** registriert (23-dim: pro Arm XYZ(3)+6D-Rotation(6)+
Greifer-Skalar(1) x2 + 3 Taille-Gelenke). Das ist aber nur die GENERISCHE Default-Config
fuer diesen Datensatznamen im Repo, nicht zwangslaeufig das, was der tatsaechlich genutzte
Checkpoint gelernt hat.

**Vorabklaerung, Runde 2 (echte Metadaten des tatsaechlich verwendeten LeRobot-Datensatzes,
vom Nutzer bereitgestellt):** Die `info.json`/Feature-Metadaten des Trainingsdatensatzes
zeigen eindeutig:

```
observation.state: shape [28], names:
  {Left,Right}ShoulderPitch, {Left,Right}ShoulderRoll, {Left,Right}ShoulderYaw,
  {Left,Right}Elbow, {Left,Right}WristRoll, {Left,Right}WristPitch, {Left,Right}WristYaw,
  LeftHandThumb0, LeftHandThumb1, LeftHandThumb2, LeftHandMiddle0, LeftHandMiddle1,
  LeftHandIndex0, LeftHandIndex1,
  RightHandThumb0, RightHandThumb1, RightHandThumb2, RightHandIndex0, RightHandIndex1,
  RightHandMiddle0, RightHandMiddle1
action: shape [28], IDENTISCHE Namensliste
observation.images.{cam_left_high, cam_right_high, cam_left_wrist, cam_right_wrist}
```

Das ist **echtes rohes Joint-Space** (28 = 14 Arm- + 14 Dex3-Fingergelenke), **keine**
Taille, **kein** EE-Pose-Format -- bestaetigt damit die urspruengliche Nutzer-Aussage fuer
diesen konkreten Checkpoint (der generische `EE_R6_G1`-Eintrag im Repo-Code betrifft
offenbar ein anderes/das Standard-Modality-Preset, nicht diesen Datensatz). Kamera-Namen
stimmen 1:1 mit denen dieser Env ueberein, keine Anpassung noetig.

Zusaetzlich bestaetigt durch den bereits vom Nutzer gebauten Inferenz-Client
(`/home/omniverse-2/unifolm-vla/unifolm-vla/deployment/isaaclab_bridge/vla_dds_client.py`):
schreibt das unnormalisierte Modell-Aktions-Array **direkt und ohne jede IK** in
`ARM_JOINT_TARGET_INDICES = range(15, 29)` (die 14 Arm-Motor-Indizes im 29-DOF
G1-Motor-Array) der DDS-`MotorCmd`-Struktur -- kein Pink/Pinocchio-Import im gesamten
Client. (Die Dex3-Fingergelenke laufen dort ueber einen separaten
`--enable_dex1_dds`-Kanal, nicht ueber diesen Aktionsvektor -- fuer unsere IsaacLab-Env
macht das keinen Unterschied, da wir ohnehin ALLE 28 Gelenke gemeinsam als eine
`JointPositionActionCfg` modellieren.)

**WICHTIG -- Reihenfolge, nicht nur Dimensionalitaet:** Die Reihenfolge im Datensatz ist
bei den Dex3-Fingergelenken ASYMMETRISCH zwischen links und rechts (links:
Thumb->Middle->Index, rechts: Thumb->Index->Middle). `JointPositionActionCfg` sortiert das
Aktions-/Beobachtungsarray standardmaessig (`preserve_order=False`) nach der INTERNEN
Artikulations-Gelenkreihenfolge, nicht nach der Reihenfolge der uebergebenen
`joint_names`-Liste (siehe `isaaclab.utils.string.resolve_matching_names`). Deshalb in der
Datei explizit `preserve_order=True` gesetzt und `DEX3_ARM_HAND_JOINT_NAMES_ORDERED` als
literale (nicht gruppierende) Gelenknamen-Liste in exakt der Datensatz-Reihenfolge definiert
-- inkl. einer eigenen `get_ordered_joint_pos`-Beobachtungsfunktion (Namens-Index-Lookup)
fuer die spiegelbildliche Garantie auf der Beobachtungsseite.

**Aufbau:** identisch zur Pink-IK-CubeStack-Env (gleiche Szene/Kameras/Wuerfel/
Erfolgscheck, per Teleop bereits bestaetigt funktionsfaehig), nur:
- `ActionsCfg`: `JointPositionActionCfg` ueber `DEX3_ARM_HAND_JOINT_NAMES_ORDERED`
  (14 Arm- + 14 Dex3-Fingergelenke = 28, **keine Taille**), `preserve_order=True`.
- `ObservationsCfg`: neue `arm_hand_joint_pos_ordered` (28-dim, exakte Datensatz-Reihenfolge)
  ersetzt die bisherigen `robot_joint_pos`/`hand_joint_state`-Terms fuer diesen Zweck.
- Kein XR-Anker/`teleop_devices` (wie bei der Parallelgreifer-Joint-Space-Variante) --
  ohne Pink-IK gibt es keinen eingebauten Retargeter fuer Hand-Tracking->Gelenkwinkel.

**Status:** Datei + Registrierung vorbereitet, Aktions-/Beobachtungsformat gegen die echten
Trainingsdatensatz-Metadaten verifiziert, **per Teleop-Testlauf erfolgreich als
konstruierbar bestaetigt** (Szene/Roboter/alle 28 Gelenknamen/Kameras loesen fehlerfrei
auf).

**Fund beim Testlauf (kein Bug, erwartetes Verhalten):**
`--task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace --teleop_device
keyboard --enable_pinocchio --headless` startet die Env erfolgreich; beim ersten
Simulationsschritt kommt dann:
```
ERROR: Error during simulation step: Invalid action shape, expected: 28, received: 7.
```
Grund: `Se3Keyboard` ist ein generisches 6-DoF-Pose-Delta+Greifer-Geraet (7 Werte), diese
Env erwartet aber 28 rohe Gelenkwinkel -- es gibt (wie im Docstring vermerkt) bewusst
keinen Retargeter, der Keyboard/Hand-Tracking auf 28 Einzelgelenke abbildet. Kein Fehler
in der Env, nur eine Einschraenkung fuer interaktives Testen; fuer skriptgesteuerte
UniFolm-VLA-Auswertung (28-dim Aktionsvektor direkt per `env.step()`) irrelevant.

**Nebenbefund (Doku-Fix):** `teleop_se3_agent.py` importiert
`isaaclab_tasks.manager_based.locomanipulation.pick_place` (und damit alle
`gym.register(...)`-Aufrufe dieses Moduls, inkl. aller 3 CubeStack-Varianten) NUR wenn
`--enable_pinocchio` gesetzt ist -- unabhaengig davon, ob die gewaehlte Task selbst
Pinocchio/Pink-IK braucht. Ohne dieses Flag meldet gym "Environment ... doesn't exist",
auch wenn Datei und Registrierung korrekt sind. Daher `--enable_pinocchio` bei ALLEN
pick_place-Tasks in diesem Ordner mitgeben, auch bei den beiden Joint-Space-Varianten ohne
eigene Pinocchio-Abhaengigkeit.

## Closed-Loop-Bruecke: IsaacLab (CubeStack-JointSpace) <-> UniFolm-VLA

Ziel: `CubeStack_JointSpace` (plain IsaacLab, `isaac-lab-base`-Container) per Closed-Loop
mit dem bereits fuer Dex1 funktionierenden UniFolm-VLA-Setup testen -- OHNE
`unitree_sim_isaaclab`s `sim_main.py` zu benutzen (das ist ein eigenes Skript fuer
`unitree_sim_isaaclab`s eigenes Task-/Env-Framework, laeuft in einem anderen Container und
ist nicht auf einen `isaaclab_tasks`-Gym-Env wie unseren anwendbar).

**Wichtige Erkenntnis:** Trotz des "dds" in den Namen (`dds_robot_cmd`,
`isaac_dex3_state`, ...) verwendet `vla_dds_client.py` **keine echte DDS-Kommunikation**
(kein `unitree_sdk2py` `ChannelPublisher`/`Subscriber` im gesamten Skript) -- es ist reines
`multiprocessing.shared_memory`, also derselbe Grundansatz wie der eingebettete ZMQ-Client
fuer GR00T (`eval_referenz_korb_ausladen.py`), nur mit SHM statt ZMQ als Transport. Das
erlaubt, `vla_dds_client.py` **unveraendert** gegen eine eigene Bruecke laufen zu lassen,
die statt `sim_main.py` unseren IsaacLab-Env bedient.

### 3 reale Bugs in `vla_dds_client.py` gefunden und behoben (relevant fuer JEDE Nutzung
dieses Clients, nicht nur fuer unsere neue Bruecke)

1. **Hand-Kommandos wurden nirgends angewendet.** `action_provider_dds.py` liest
   Handgelenk-Ziele aus einem SEPARATEN SHM-Kanal (`isaac_dex3_cmd`, ueber
   `Dex3DDS.get_hand_commands()`), komplett unabhaengig von `dds_robot_cmd` (nur die 29
   Koerpermotoren). `vla_dds_client.py` schrieb nie dorthin -- die letzten 14 Werte eines
   28-dim Modell-Outputs gingen ins Leere. Behoben: neuer `dex3_cmd_shm`-Kanal +
   `_build_dex3_hand_cmd()`.
2. **Reihenfolge rechte Hand asymmetrisch.** Trainingsdatensatz: rechts
   Thumb->Index->Middle. Sim (`action_provider_dds.py::right_hand_joint_mapping`,
   `dex3_state.py::get_robot_girl_joint_names`): rechts Thumb->Middle->Index (wie links,
   symmetrisch). `_RIGHT_HAND_DATASET_TO_SIM_ORDER = [0,1,2,5,6,3,4]` konvertiert -- diese
   Permutation ist selbstinvers (vertauscht nur 2 gleich grosse Bloecke), daher fuer beide
   Richtungen (Schreiben der Aktion, Lesen des Zustands) wiederverwendbar.
3. **Proprio-Konstruktion falsch.** `isaac_robot_state`'s `joint_positions` enthaelt nur
   die 29 Koerpermotoren (Beine+Taille+Arme, KEINE Hand). Der generische
   `_compose_proprio`-Fallback konkateniert das blind mit velocities/torques/imu und
   schneidet auf `proprio_dim`(28) zurecht -- voellig andere Verteilung als das trainierte
   14-Arm+14-Hand-Proprio. Behoben: neue `_compose_dex3_proprio()`, die die Arm-Werte aus
   `isaac_robot_state` UND die Handwerte separat aus `isaac_dex3_state` liest und in der
   korrekten Datensatz-Reihenfolge zusammensetzt (mit Warteschleife, falls der Hand-State
   noch nicht publiziert wurde).

Zusaetzlich `tools/shared_memory_utils.py::MultiImageReader` gefixt: `read_images()` hatte
`image_names = ['head','left','right']` FEST verdrahtet, unabhaengig vom `--image_order`-
Flag -- `--image_order cam_left_high ...` haette also nie etwas gefunden. Jetzt
konfigurierbar per Konstruktor-Parameter (Default unveraendert fuer Abwaertskompatibilitaet
mit dem bestehenden Dex1-Setup).

### Neue Datei: `scripts/tools/cubestack_jointspace_unifolm_vla_bridge.py`

Spielt fuer unseren IsaacLab-Env dieselbe Rolle wie `sim_main.py` fuer
unitree_sim_isaaclab: steppt den Env, publiziert Kamerabilder + Roboter-/Hand-Zustand in
dieselben SHM-Segmente, die `vla_dds_client.py` bereits erwartet, und wendet die
zurückgeschriebene Aktion an. Eigene, abhaengigkeitsfreie Nachimplementierung von
`SharedMemoryManager`/Bild-Header (kein Import aus unitree_sim_isaaclab noetig, da dieser
Pfad im `isaac-lab-base`-Container nicht gemountet ist) -- selbes Wire-Format, daher 1:1
kompatibel. Kamera-Encoding ist RAW RGB (kein JPEG/cv2), daher beim Client
`--camera_color_space rgb` verwenden (nicht `bgr`).

**Terminal 1 (im `isaac-lab-base`-Container):**
```bash
./isaaclab.sh -p scripts/tools/cubestack_jointspace_unifolm_vla_bridge.py \
    --task Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-JointSpace \
    --device cpu --enable_cameras --enable_pinocchio --headless
```

**Terminal 2 (Host, `conda activate unifolm-vla`):**
```bash
cd /home/omniverse-2/unifolm-vla/unifolm-vla
python deployment/isaaclab_bridge/vla_dds_client.py \
    --ckpt_path /home/omniverse-2/UnifoLM-VLA-Base/checkpoints/pytorch_model.pt \
    --vlm_pretrained_path unitreerobotics/UnifoLM-VLM-Base \
    --instruction "stack the red cube on the yellow cube, then the green cube on top" \
    --device cuda --camera_color_space rgb \
    --image_order cam_left_high cam_right_high cam_left_wrist cam_right_wrist \
    --log_actions
```

**Status:** Alle Teile geschrieben und syntaktisch geprueft, **noch NICHT end-to-end
getestet** -- naechster Schritt ist ein echter Lauf, um Restfehler (SHM-Groessen,
Timing/Race-Conditions zwischen Bruecke und Client, Joint-Limit-Clamping etc.) zu finden.

## Naechste Schritte

1. Einmal headless starten, sowohl fuer
   `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack` (Dex3) als auch fuer
   `Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-CubeStack-ParallelGripper` (Dex1), um
   Szenenaufbau, Kamera-Rendering, Kontaktphysik und insbesondere bei der neu portierten
   Dex1-Variante die USD-/Kinematik-Kompatibilitaet zu verifizieren.
2. Ggf. Wuerfelpositionen/Reset-Fenster gegen den tatsaechlich erreichbaren
   G1-Arbeitsbereich (fixed base) nachjustieren; bei der Dex1-Variante zusaetzlich die
   Handgelenkkamera-Offsets pruefen.
3. Demonstrationen aufnehmen (`record_demos.py`, analog zum Referenz-Workflow in
   `referenz_workflow2_no_camera_teleop_und_stitching.md`), GR00T-Fine-Tuning auf der
   Dex3-Variante und UniVLA-Fine-Tuning/-Test auf der Dex1-Variante durchfuehren.
4. Kopie von `eval_referenz_korb_ausladen.py` fuer Cube Stacking anlegen (Task-ID +
   `TASK_DESCRIPTION` austauschen, Rest des ZMQ-Eval-Protokolls unveraendert), sowohl
   fuer GR00T- als auch fuer einen UniVLA-Policy-Server.
5. Ergebnisse (Open-/Closed-Loop Success Rate) vergleichend fuer GR00T (Dex3) vs. UniVLA
   (Parallelgreifer) dokumentieren.
