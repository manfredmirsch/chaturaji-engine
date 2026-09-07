#!/usr/bin/env python3
"""Packt die chess.com-Partien für den Upload ins Release `nnue-state`.

Die Dateien aus `game_data/` enthalten viel, was das Training nie liest —
Avatare, Chat, IP-Hashes, Ratings, Zeitkontrolle. Gebraucht werden nur die
Zugfolge und der Endstand. Das Weglassen ist kein Sparen um des Sparens
willen: die Weboberfläche von GitHub nimmt nur Dateien bis 25 MB, und das
vollständige Archiv liegt darüber.

    vollständig, gzip   32 MB   ← zu groß für den Upload im Browser
    reduziert, gzip     20,5 MB ← passt, und `.gz` nimmt GitHub an
    reduziert, xz       13 MB   ← kleiner, aber `.xz` wird abgelehnt

Geprüft: mit den reduzierten Dateien liefert `diagnose_pgn` dieselben Zahlen
wie mit den vollständigen — 11.558 nachspielbar, 11.555 exakte Endstände.

Aufruf:
    python3 scripts/pack-game-data.py ~/chaturaji/game_data ~/chaturaji/game_data.tar.gz
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Was der Importer tatsächlich liest (siehe crates/nnue/src/pgn_import.rs):
# `pgn4` für die Züge, `points1..4` für den Endstand, `standings` als
# Rückfallebene, wenn die Punkte fehlen.
KEEP = ("pgn4", "points1", "points2", "points3", "points4", "standings", "gameNr")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    src = pathlib.Path(sys.argv[1])
    dst = pathlib.Path(sys.argv[2])
    if not src.is_dir():
        print(f"Verzeichnis '{src}' nicht gefunden.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "game_data"
        out.mkdir()

        written = skipped = 0
        for path in sorted(src.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                skipped += 1
                continue
            slim = {k: data[k] for k in KEEP if k in data}
            if "pgn4" not in slim:
                skipped += 1
                continue
            (out / path.name).write_text(json.dumps(slim, separators=(",", ":")))
            written += 1

        if written == 0:
            print("Keine verwertbaren Partien gefunden.")
            return 1

        # Kompression an der Endung ablesen; `tar` erkennt sie beim Entpacken
        # von selbst, der Workflow braucht dafür nichts zu wissen.
        flag = "J" if dst.suffix == ".xz" else "z"
        subprocess.run(
            ["tar", f"c{flag}f", str(dst.resolve()), "game_data"],
            cwd=tmp, check=True,
        )

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{written} Partien gepackt ({skipped} übersprungen)")
    print(f"{dst}  —  {size_mb:.1f} MB")
    if size_mb > 25:
        print("Achtung: über 25 MB, der Browser-Upload bei GitHub lehnt das ab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
