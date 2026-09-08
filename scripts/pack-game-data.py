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

Partien mit abweichendem Königswert werden aussortiert — siehe KING_VALUE_RE.

Aufruf:
    python3 scripts/pack-game-data.py <quelle> <archiv.tar.gz> [--keep <verz>]

`--keep` legt die abgespeckten Dateien zusätzlich als Verzeichnis ab, statt sie
nur im Archiv zu haben.
"""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

# Was der Importer tatsächlich liest (siehe crates/nnue/src/pgn_import.rs):
# `pgn4` für die Züge, `points1..4` für den Endstand, `standings` als
# Rückfallebene, wenn die Punkte fehlen.
KEEP = ("pgn4", "points1", "points2", "points3", "points4", "standings", "gameNr")

# chess.com lässt den Königswert einstellen. Der Regelsatz der Engine rechnet
# fest mit 3 (`PieceKind::King::capture_value`), und in Partien mit einer
# anderen Einstellung läuft der Punktestand ab dem ersten Königsschlag
# auseinander — je geschlagenem König um die Differenz. Solche Partien gehören
# nicht ins Training: ihr Punktestand geht als `dense_features` ins Netz ein,
# und das Netz sähe Stände, die es nach den geltenden Regeln nie geben kann.
#
# Erkennbar ist die Einstellung nur an der Ansage im Chat — und nur in den
# Rohdaten. Nach dem Abspecken ist sie weg, das Aussortieren muss also hier
# geschehen.
STANDARD_KING_VALUE = 3
KING_VALUE_RE = re.compile(r"Checkmates/kings = \+(\d+) points")


def king_value(game):
    """Der eingestellte Königswert laut Chat, sonst None."""
    for entry in game.get("chat", []):
        m = KING_VALUE_RE.search(entry.get("message", "") or "")
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    keep_dir = None
    for i, a in enumerate(sys.argv):
        if a == "--keep" and i + 1 < len(sys.argv):
            keep_dir = pathlib.Path(sys.argv[i + 1])
            argv = [x for x in argv if x != sys.argv[i + 1]]
    if len(argv) != 2:
        print(__doc__)
        return 2

    src = pathlib.Path(argv[0])
    dst = pathlib.Path(argv[1])
    if not src.is_dir():
        print(f"Verzeichnis '{src}' nicht gefunden.")
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "game_data"
        out.mkdir()

        written = skipped = odd_rules = 0
        for path in sorted(src.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (OSError, ValueError):
                skipped += 1
                continue
            kv = king_value(data)
            if kv is not None and kv != STANDARD_KING_VALUE:
                print(f"  {path.name}: Königswert {kv} statt "
                      f"{STANDARD_KING_VALUE} — aussortiert")
                odd_rules += 1
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

        if keep_dir is not None:
            keep_dir.mkdir(parents=True, exist_ok=True)
            for old in keep_dir.glob("*.json"):
                old.unlink()
            for f in out.glob("*.json"):
                (keep_dir / f.name).write_text(f.read_text())
            print(f"Abgespeckte Dateien auch in {keep_dir}/")

    size_mb = dst.stat().st_size / 1024 / 1024
    print(f"{written} Partien gepackt ({skipped} übersprungen, "
          f"{odd_rules} mit abweichendem Königswert)")
    print(f"{dst}  —  {size_mb:.1f} MB")
    if size_mb > 25:
        print("Achtung: über 25 MB, der Browser-Upload bei GitHub lehnt das ab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
