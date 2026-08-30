#!/usr/bin/env python3
"""
camera_geometry.py — Kameraposen, Intrinsics und das Pinhole-Modell dazu.

Herausgelöst aus ``g1_dex3_cfg.py`` (2026-08-17), weil der Inhalt **kein Isaac Lab
braucht**: es sind Posen, Brennweiten und Projektionsrechnung, also numpy. Nur so lässt
sich das Modell außerhalb des Containers prüfen — und die Prüfung ist der Punkt. Der
Kamera-Pose-Bug aus Lauf 13 (2026-08-08, 95,6° zwischen USD-Stage und ``cam.data``) kostete
drei Sim-Läufe, weil niemand Konfiguration gegen gerendertes Bild gehalten hat.

``g1_dex3_cfg.py`` importiert von hier und exportiert die Namen weiter; für die Env ändert
sich nichts.

WARNUNG ZUR GÜLTIGKEIT DES MODELLS
──────────────────────────────────
Die Posen hier sind aus den Dataset-Frames **rekonstruiert**, nicht kalibriert (14 Overlay-
Iterationen, s. Kommentare unten). Und ob Isaac mit der konfigurierten Pose auch rendert,
ist eine eigene Frage — genau die ging in Lauf 13 schief. Deshalb gilt:

    Bevor eine Rückprojektion (Bild → Weltkoordinate) irgendwo verwendet wird, muss die
    Vorwärtsrichtung gegen ein GERENDERTES Bild geprüft sein: bekannte Würfelposition
    projizieren, im Sim-Bild den Würfel finden, Abstand messen.

``extract_block_layout.py detect --expect ...`` macht genau das.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def look_at_world_quat(eye, target, world_up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) für eine Kamera in Isaac-Lab-``convention="world"``.

    In dieser Konvention ist die **Blickachse +X** und **oben +Z**. Die Funktion
    richtet die Kamera so aus, dass sie von ``eye`` auf ``target`` blickt — damit lassen
    sich die Posen über anschauliche Punkte statt undurchsichtiger Quaternionen tunen.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(world_up, dtype=float)

    x = target - eye                      # Blickrichtung = Kamera-+X
    x /= np.linalg.norm(x)
    z = up - np.dot(up, x) * x            # Kamera-oben (+Z) = Welt-oben ⟂ Blickachse
    if np.linalg.norm(z) < 1e-6:          # Blick exakt vertikal → Ersatz-up
        z = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)                    # rechtshändig: x × y = z

    R = np.column_stack([x, y, z])        # Spalten = Kamera-Achsen im Weltframe
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * w)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * w)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * w)
    return (float(w), float(qx), float(qy), float(qz))


def quat_to_matrix(q) -> np.ndarray:
    """(w, x, y, z) → Rotationsmatrix, deren SPALTEN die Kamera-Achsen im Weltframe sind.

    Die Umkehrung von ``look_at_world_quat``: dort wird R aus den Achsen gebaut und in ein
    Quaternion umgerechnet, hier zurück. Spalte 0 ist damit die Blickachse, Spalte 2 „oben".
    """
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class G1Dex3CameraCfg:
    """Kamera-Positionen und Intrinsics — möglichst nah an den Trainingsdaten."""

    # Auflösung der Kamerabilder (anpassen auf die tatsächliche Dataset-Auflösung)
    width: int = 640
    height: int = 480

    # Horizontaler FOV in Grad. Overlay-Iteration: 69° zu eng (Sim zu nah), 90° zu weit
    # (Szene zu sparse, Hände winzig). 75° als Kompromiss.
    #
    # ACHTUNG, 2026-08-08: Dieser Wert war reine Dokumentation — er stand hier, wurde aber
    # nirgends gelesen. Die Kameras liefen mit der fest eingetragenen `focal_length=24.0`
    # aus `g1_dex3_blockstack_env.py`, also mit 47,2° statt 75°. Die Sim-Bilder waren damit
    # dauerhaft zu eng, und die damals als „zu eng" verworfenen 69° waren nie im Bild.
    # `focal_high`/`focal_wrist`/`focal_scene` unten rechnen die Werte jetzt tatsächlich um.
    hfov_deg: float = 75.0
    # Handgelenks- und Szenenkamera behalten vorerst ihr bisheriges Sichtfeld (focal 18.0 =
    # 60,4°) — für die ist keine Overlay-Iteration dokumentiert, ein Wechsel wäre geraten.
    hfov_wrist_deg: float = 60.4
    hfov_scene_deg: float = 60.4

    # Sensorbreite in mm. Isaac Lab leitet die vertikale Apertur aus dem Seitenverhältnis ab.
    horizontal_aperture_mm: float = 20.955

    # Kamera-Posen (pos in Meter, rot als (w, x, y, z) Quaternion, World-Frame)
    # WICHTIG: Aus den Dataset-Videos rekonstruieren! Dies sind Schätzwerte.
    cam_left_high: dict = None
    cam_right_high: dict = None
    # Wrist-Kameras werden relativ zum Wrist-Link definiert (lokal)
    cam_left_wrist_local: dict = None
    cam_right_wrist_local: dict = None
    # Szenen-Übersichtskamera — NUR fürs Video, NICHT Policy-Observation
    cam_scene: dict = None

    # Aus hfov_*_deg berechnet (siehe __post_init__) — die Env liest diese Werte.
    focal_high: float = None
    focal_wrist: float = None
    focal_scene: float = None

    def __post_init__(self):
        # Brennweiten aus dem gewünschten Sichtfeld, damit die hfov-Angaben oben wirken.
        def focal(hfov_deg: float) -> float:
            return self.horizontal_aperture_mm / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))

        self.focal_high = float(focal(self.hfov_deg))       # 75.0° -> 13.65 mm (vorher 24.0)
        self.focal_wrist = float(focal(self.hfov_wrist_deg))  # 60.4° -> 18.0 mm (unverändert)
        self.focal_scene = float(focal(self.hfov_scene_deg))  # 60.4° -> 18.0 mm (unverändert)
        # Externe Kameras: Kopf-Stereo-Paar, blickt von vorne-oben NACH UNTEN auf den Tisch.
        # Rekonstruiert aus den Dataset-Frames (Simulation/camera_reference/dataset_cam_*_high.png):
        # beide Hände kommen von unten ins Bild, Tisch füllt die unteren ~2/3.
        # Rotation per Look-at auf die Tischmitte — frühere (0.924,-0.383,0,0) war eine reine
        # Roll-Drehung um die +X-Blickachse (Bild verkippt, kein Pitch) und zeigte auf den Boden.
        #
        # Overlay-Iteration 6: Reale Referenz (dataset_cam_*_high.png) zeigt eindeutig einen
        # FLACHEN Vorwärts-Blick vom Kopf: beide Hände kommen von unten-links/rechts ins Bild,
        # Finger zeigen nach oben-vorne, Tisch füllt die Mitte. KEINE steile Top-Down-Sicht.
        # Iter 5 (target z=0.80, steil nach unten) ließ die Arme aus dem Bild fallen.
        # Fix: eye zentriert auf Kopfhöhe (z=1.45, x=0), Target weit nach VORNE (x=0.55) auf
        # Tischhöhe → flacher Pitch (~46°), Arme reichen von unten ins Bild, Tisch in der Mitte.
        #
        # Iteration 13 (2026-08-08), nachdem die Kameras überhaupt wieder rendern. Der
        # Overlay gegen Simulation/camera_reference/ zeigte drei Abweichungen, alle drei
        # gegen eine unabhängige Quelle geprüft statt geschätzt:
        #
        # a) MONTAGEPUNKT. Die Kameras standen auf z=1.45 und y=±0.08 — also NEBEN und ÜBER
        #    dem Kopf, weshalb der eigene Kopf ein Drittel des Bildes verdeckte (Lauf 16).
        #    Der echte G1 trägt seine Kopfkamera laut URDF im `d435_link`, pelvis-relativ
        #    (0.0537, 0.0175, 0.4739); das Pelvis sitzt env-lokal auf z=0.85, macht
        #    (0.0537, 0.0175, 1.3239).
        # b) STEREO-BASIS. ±0.08 wären 16 cm Basis. Aus den beiden Referenzbildern gemessen:
        #    Querversatz 40 px (SAD-Minimum über die Tischplatte), Würfelkantenlänge 40–46 px
        #    bei bekannten 5 cm → Motivabstand ~0.57 m → Basis ~4.7 cm. Der Wert hängt nicht
        #    am angenommenen FOV, weil Abstand und Winkel gemeinsam mitskalieren. Passt zu
        #    den 50 mm einer RealSense D435, also ±0.025.
        # c) PARALLEL STATT KONVERGENT. Beide Kameras auf EIN gemeinsames Ziel zu richten
        #    erzeugte ±8,3° Gierwinkel und damit die schräge Tischkante im Bild. Ein reales
        #    Stereopaar blickt parallel — deshalb bekommt jede Kamera ihr Ziel auf der
        #    eigenen y-Linie.
        #
        # Zielpunkt = Mitte des Würfel-Spawnbereichs (x≈0.34, z≈0.915). Das ergibt 55°
        # Neigung; damit liegt die Tischhinterkante bei ~5 % und die Vorderkante bei ~99 %
        # der Bildhöhe, der Tisch also vollständig im Bild wie in der Referenz, und die
        # Würfel stehen mittig. Beide Hände sind dabei 21,5° von der Blickachse entfernt
        # und damit deutlich innerhalb des 75°×59,9°-Sichtfelds.
        #
        # Iteration 14 (Lauf 17): Die Basis wird um y=0 zentriert statt um die y=0.0175 des
        # `d435_link`. Grund: in der Reset-Pose stehen die Handgelenke fast symmetrisch
        # (y=+0.158 / −0.144), und im Referenzbild liegen beide Hände symmetrisch um die
        # Bildmitte — die reale Kamera sitzt also auf der Mittellinie. Das `d435_link` ist
        # der Montageflansch eines Moduls, nicht der Mittelpunkt zwischen zwei Bildsensoren.
        # x und z bleiben beim URDF-Wert, die sind eindeutig.
        high_target_x, high_target_z = 0.34, 0.915
        left_high_eye  = (0.0537,  0.025, 1.3239)   # halbe Stereobasis links der Mittellinie
        right_high_eye = (0.0537, -0.025, 1.3239)   # halbe Stereobasis rechts der Mittellinie
        self.cam_left_high = {
            "pos": left_high_eye,
            "rot": look_at_world_quat(
                left_high_eye, (high_target_x, left_high_eye[1], high_target_z)),
        }
        self.cam_right_high = {
            "pos": right_high_eye,
            "rot": look_at_world_quat(
                right_high_eye, (high_target_x, right_high_eye[1], high_target_z)),
        }
        # Wrist-Kameras: am jeweiligen Wrist-Yaw-Link montiert (convention="world", Link-Frame).
        # Die Hand ragt entlang +X (Palm-Joint bei x=0.0415, Finger weiter bei +X). Look-at von
        # hinter/über dem Wrist-Origin (raus aus dem Palm-Mesh) auf die Fingerspitzen → die Kamera
        # zeigt garantiert auf die Hand (vorher: pos=(0.05,…) steckte IM Palm-Mesh → nur Grau;
        # rechte Cam zudem falsch herum (-X)). Roll/Feinframing nach Render-Vergleich justieren.
        # Overlay-Iteration 6: KORREKTUR des früheren Z-Flips. Die reale Referenz
        # (dataset_cam_left_wrist.png) zeigt die Kamera von OBEN-HINTEN nach UNTEN-VORNE über
        # die Finger auf den Tisch blickend — das dunkle Gehäuse oben im realen Bild ist der
        # Unterarm (= weißer Connector in der Sim), der nur das obere Drittel einnimmt.
        # Der vorherige Flip nach -Z (Kamera unter dem Wrist) war falsch und ließ den Connector
        # das ganze Bild füllen. Fix: eye wieder ÜBER den Wrist (+Z), leicht hinter den Knöcheln
        # (-X), Blick nach vorne-unten (+X, -Z) auf Fingerspitzen + Tisch.
        # Iter 7: Target leicht angehoben (-0.12 → -0.06), damit der Blick die Tischfläche
        # statt des Bodengitters dahinter trifft (Iter 6 pitchte minimal über die Tischkante).
        # Iter 10: Kamera ein Stück entlang der Handachse (+X) Richtung Finger geschoben
        # (eye -0.08 → 0.0, target 0.22 → 0.30). Iter 12: war etwas zu weit vorne, zurück auf
        # die Mitte zwischen Iter 9 und 10 (eye -0.04, target 0.26). Blickrichtung/Roll bleiben.
        wrist_eye = (-0.04, 0.0, 0.10)
        left_wrist_target  = (0.26, 0.0, -0.06)
        right_wrist_target = (0.26, 0.0, -0.06)
        # Iter 8 — ROLL-Korrektur: Pitch/Blickrichtung stimmten, aber die Hand stand im Sim
        # VERTIKAL, im Real liegt sie HORIZONTAL (Finger nach rechts statt nach unten) → ~90°
        # Roll-Versatz. Ursache: look_at nutzte default world_up=(0,0,1), was im rotierten
        # Wrist-Link-Frame den falschen Roll erzeugt. Fix: world_up auf die laterale Link-Achse
        # (+Y) legen, damit die Hand horizontal im Bild liegt.
        # Iter 9: linke Cam mit +Y war korrekt (Finger horizontal nach rechts, deckt sich mit
        # Real). Rechte Cam mit -Y war um 180° verdreht (Finger nach links statt oben-rechts) →
        # der rechte Wrist-Link nutzt DIESELBE Frame-Konvention wie links, kein gespiegeltes
        # Frame. Daher beide Cams +Y. Restlicher ~45°-Diagonal-Versatz rechts = reale Pose-
        # Differenz (asymmetrische Init-Pose), ggf. später per Z-Komponente im Up feinjustieren.
        # Iter 12: rechte Cam wieder auf (0,1,0) wie im letzten gerenderten Run — der in Iter 11
        # versuchte -Z-Tilt wurde nie validiert und auf Wunsch zurückgenommen. Beide Cams +Y.
        left_wrist_up  = (0.0, 1.0, 0.0)
        right_wrist_up = (0.0, 1.0, 0.0)
        self.cam_left_wrist_local = {
            "pos": wrist_eye,
            "rot": look_at_world_quat(wrist_eye, left_wrist_target, world_up=left_wrist_up),
        }
        self.cam_right_wrist_local = {
            "pos": wrist_eye,
            "rot": look_at_world_quat(wrist_eye, right_wrist_target, world_up=right_wrist_up),
        }

        # Szenen-Übersichtskamera (NUR fürs aufgenommene Video): zeigt die GANZE Szene —
        # Roboter (Pelvis z=0.85, Kopf ~1.4) + Tisch (x=0.5) — von schräg vorne-seitlich-oben.
        # Weltfest. eye in +X (vor dem Tisch), +Y (seitlich), +Z (oben); Blick zurück auf die Mitte.
        scene_eye = (1.8, 1.6, 1.7)
        scene_target = (0.30, 0.0, 0.80)
        self.cam_scene = {"pos": scene_eye, "rot": look_at_world_quat(scene_eye, scene_target)}


CAMERA_CFG = G1Dex3CameraCfg()


# ---------------------------------------------------------------------------
# Pinhole-Modell — Projektion und Rückprojektion
# ---------------------------------------------------------------------------

class PinholeCamera:
    """Projektion und Rückprojektion für eine der weltfesten Kameras.

    Bildachsen in Isaac-Lab-``convention="world"``: Blick = +X, oben = +Z, links = +Y.
    Bild-RECHTS ist damit −Y und Bild-UNTEN ist −Z — die beiden Vorzeichen sind der
    ganze Trick, und sie sind der Grund, warum das Modell gegen ein gerendertes Bild
    geprüft werden muss statt nur gegen den eigenen Verstand.

    Pixel werden als Index gezählt (0 … W−1); die Mitte von Pixel u liegt bei u + 0,5.
    """

    def __init__(self, eye, quat, focal_mm: float, aperture_mm: float,
                 width: int, height: int):
        self.eye = np.asarray(eye, dtype=float)
        self.R = quat_to_matrix(quat)            # Spalten: [Blick, links, oben]
        self.width, self.height = int(width), int(height)
        # Isaac Lab leitet die vertikale Apertur aus dem Seitenverhältnis ab, die Pixel sind
        # also quadratisch und f_px gilt für beide Achsen.
        self.f_px = float(focal_mm) / float(aperture_mm) * self.width
        self.cx, self.cy = self.width / 2.0, self.height / 2.0

    @classmethod
    def from_cfg(cls, name: str, cfg: G1Dex3CameraCfg = CAMERA_CFG) -> "PinholeCamera":
        """Eine der weltfesten Kameras aus der Konfiguration bauen.

        Nur die drei weltfesten sind hier zulässig. Die Wrist-Kameras hängen an einem Link,
        ihre Weltpose ändert sich mit jeder Roboterpose — die kann nur die laufende Sim
        liefern, nicht eine Konfigurationsdatei.
        """
        focal = {"cam_left_high": cfg.focal_high, "cam_right_high": cfg.focal_high,
                 "cam_scene": cfg.focal_scene}
        if name not in focal:
            raise ValueError(
                f"{name} ist nicht weltfest (oder unbekannt). Verfügbar: {sorted(focal)}. "
                "Wrist-Kameras bewegen sich mit dem Link und brauchen die Pose aus der Sim."
            )
        pose = getattr(cfg, name)
        return cls(pose["pos"], pose["rot"], focal[name], cfg.horizontal_aperture_mm,
                   cfg.width, cfg.height)

    def project(self, points_world) -> np.ndarray:
        """Weltpunkte (…,3) → Pixel (…,2) als (u, v). Punkte hinter der Kamera: NaN."""
        p = np.atleast_2d(np.asarray(points_world, dtype=float))
        cam = (p - self.eye) @ self.R          # Komponenten entlang [Blick, links, oben]
        fwd = cam[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f_px * (-cam[:, 1] / fwd) - 0.5
            v = self.cy + self.f_px * (-cam[:, 2] / fwd) - 0.5
        out = np.stack([u, v], axis=-1)
        out[fwd <= 1e-6] = np.nan
        return out.reshape(np.shape(points_world)[:-1] + (2,))

    def ray(self, u: float, v: float) -> np.ndarray:
        """Blickstrahl (Einheitsvektor im Weltframe) durch die Mitte von Pixel (u, v)."""
        d_right = ((u + 0.5) - self.cx) / self.f_px
        d_down = ((v + 0.5) - self.cy) / self.f_px
        d_cam = np.array([1.0, -d_right, -d_down])   # rechts = −links, unten = −oben
        d = self.R @ d_cam
        return d / np.linalg.norm(d)

    def backproject_to_plane(self, u: float, v: float, z_plane: float) -> np.ndarray:
        """Pixel (u, v) auf die waagerechte Ebene z = ``z_plane`` schneiden → (x, y, z)."""
        d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            raise ValueError(f"Strahl durch ({u}, {v}) ist parallel zur Ebene z={z_plane}.")
        t = (z_plane - self.eye[2]) / d[2]
        if t <= 0:
            raise ValueError(
                f"Ebene z={z_plane} liegt HINTER der Kamera für Pixel ({u:.0f}, {v:.0f})."
            )
        return self.eye + t * d

    def plane_scale_cm_per_px(self, u: float, v: float, z_plane: float) -> float:
        """Wie viele cm auf der Ebene ein Pixel an dieser Stelle abdeckt.

        Damit werden Pixel-Residuen in cm lesbar. Numerisch statt analytisch, weil die
        Skala über das Bild variiert (schräger Blick) und ein Nachbarpixel das ehrlich
        abbildet.
        """
        a = self.backproject_to_plane(u, v, z_plane)
        b = self.backproject_to_plane(u + 1.0, v, z_plane)
        return float(np.linalg.norm(b - a) * 100.0)
