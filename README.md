# Chaturaji Engine

Rust + WASM Analyse- und Trainingstool für **Chaturaji** (chess.com-Variante) –
das antike indische 4-Spieler-Schach mit neuronalem Netz und TD(λ)-Training.

---

## Schnellstart

### 1. Voraussetzungen installieren

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
echo 'source "$HOME/.cargo/env"' >> ~/.bashrc

# WASM-Target + wasm-pack
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Node.js (für den Dev-Server)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install --lts
```

### 2. Unit-Tests

```bash
cargo test -p chaturaji-core
cargo test -p chaturaji-engine
cargo test -p chaturaji-trainer
```

### 3. Web-Frontend bauen und starten

```bash
cd crates/wasm
wasm-pack build --target web --out-dir www/pkg
cd www
npx serve .          # oder: python3 -m http.server 8080
# Browser öffnen: http://localhost:3000
```

### 4. Trainer starten

```bash
# Erstes Training (10 000 Partien, läuft ca. 15-30 Minuten)
cargo run --release --bin train

# Schnelltest (100 Partien)
cargo run --release --bin train -- --games 100 --log-every 10

# Gewichte für den Browser exportieren
cargo run --release --bin train -- --export weights.json

# Statistiken anzeigen
cargo run --release --bin train -- --stats
```

### 5. Gewichte in den Browser laden

1. Web-Frontend starten (Schritt 3)
2. Im Browser: **"Neuronales Netz"** aufklappen
3. **"Gewichte laden"** → `weights.json` auswählen
4. Die Netz-Bewertungsbalken erscheinen unter dem Brett

---

## Projektstruktur

```
chaturaji-engine/
├── Cargo.toml                  Workspace
└── crates/
    ├── core/                   Spiellogik (no_std-kompatibel)
    │   └── src/
    │       ├── board.rs        8x8 Board + 4×5 Bitboards
    │       ├── piece.rs        Figurtypen & 4 Farben
    │       ├── movegen.rs      Pseudo-legaler Zuggenerator
    │       ├── rules.rs        Regeleffekte (Check-Bonus, Aufgabe, Endstand)
    │       ├── score.rs        Punkteverfolgung
    │       ├── notation.rs     Algebraische Notation & PGN
    │       └── zobrist.rs      Zobrist-Hashing
    │
    ├── engine/                 Alpha-Beta-Suche
    │   └── src/
    │       ├── eval.rs         Statische Bewertung + Piece-Square-Tables
    │       ├── ordering.rs     Zugordnung (MVV-LVA + TT)
    │       ├── tt.rs           Transpositionstabelle
    │       └── search.rs       Max^n + Iterative Deepening
    │
    ├── trainer/                TD(λ) Trainer
    │   └── src/
    │       ├── features.rs     Stellungs-Features (336 Dimensionen)
    │       ├── network.rs      Neuronales Netz (336→256→128→4)
    │       ├── selfplay.rs     Self-Play mit ε-greedy-Exploration
    │       ├── td.rs           TD(λ)-Trainingsloop
    │       ├── db.rs           SQLite: Gewichte, Partien, Statistiken
    │       └── main.rs         CLI-Einstiegspunkt
    │
    └── wasm/                   Browser-Frontend
        ├── src/lib.rs          wasm_bindgen API
        └── www/
            ├── index.html
            ├── style.css
            ├── index.js        Canvas-Brett + Netz-UI
            └── pieces/         Eigene SVG-Figuren hier ablegen
                └── README.txt
```

---

## Spielregeln (chess.com-Variante)

**Zugreihenfolge**: Rot → Blau → Gelb → Grün (im Uhrzeigersinn)

| Figur    | Bewegung                        | Punkte bei Schlag |
|----------|---------------------------------|-------------------|
| Bauer    | 1 Feld vorwärts, kein Doppelzug | 1                 |
| Springer | L-Form (Standard-Schach)        | 3                 |
| König    | 1 Feld in alle 8 Richtungen     | 3                 |
| Läufer   | Diagonal (Slider)               | 5                 |
| Schiff   | Springt genau 2 Felder diagonal | 5                 |

**Startaufstellung** (vom Rand zur Mitte: Schiff, Springer, Läufer, König):
- Rot (Süden):  a1=Schiff, b1=Springer, c1=Läufer, d1=König; Bauern a2-d2
- Blau (Westen): a8=Schiff, b8=Springer, c8=Läufer, d8=König; Bauern a7-d7
- Gelb (Norden): h8=Schiff, g8=Springer, f8=Läufer, e8=König; Bauern h7-e7
- Grün (Osten):  h1=Schiff, g1=Springer, f1=Läufer, e1=König; Bauern g1-g4

**Sonderregeln**:
- Kein Schach/Matt – Könige werden einfach geschlagen
- Wer seinen König verliert, scheidet aus (Figuren inaktiv)
- **Doppelschach**: +1 Punkt; **Dreifachschach**: +5 Punkte
- Bauernumwandlung: immer zum Schiff

---

## Neuronales Netz & TD(λ)-Training

### Architektur

```
Eingabe (336)  →  Hidden 1 (256, ReLU)  →  Hidden 2 (128, ReLU)  →  Ausgabe (4, tanh)
```

- **Eingabe**: 56 kompakte Features (Figurzahlen, Pawn-Fortschritt, Zentrum, Mobilität,
  Punktestand, Spielphase, Königssicherheit, Schiff-Bestand)
  + Padding auf 336 für zukünftige Erweiterungen
- **Ausgabe**: Bewertungsvektor `[f32; 4]` in `[-1, +1]` – je ein Wert pro Spieler
- **Parameter**: ~101 000

### TD(λ)-Algorithmus

```
Für jede Self-Play-Partie:
  1. Engine spielt gegen sich selbst (ε-greedy)
  2. Rückwärts durch alle Stellungen:
       δ_t = V(s_{t+1}) - V(s_t)        ← TD-Fehler
       e_t = λ·e_{t-1} + ∇V(s_t)        ← Eligibility Trace
       Δw  = α · δ_t · e_t               ← Gewichts-Update
  3. Letzter Schritt: Target = normalisiertes Spielergebnis
```

### Trainer-Optionen

| Option | Standard | Beschreibung |
|--------|---------|--------------|
| `--games` | 10000 | Anzahl Trainingspartien |
| `--lambda` | 0.7 | TD-Lambda (0=TD(0), 1=Monte-Carlo) |
| `--lr` | 0.001 | Lernrate (wird über Zeit abgesenkt) |
| `--depth` | 2 | Alpha-Beta-Tiefe im Self-Play |
| `--save-every` | 200 | Checkpoint-Intervall |
| `--db` | chaturaji.db | SQLite-Datenbankpfad |

### SQLite-Datenbank

Tabellen:
- `network_weights` – alle Checkpoints mit Zeitstempel und Loss
- `training_stats` – Loss-Verlauf pro Epoche
- `games` – alle gespielten Self-Play-Partien (PGN + Endstand)

---

## Eigene Figurengrafiken

SVG-Dateien in `crates/wasm/www/pieces/` ablegen:

```
pieces/red-king.svg      pieces/blue-king.svg
pieces/red-bishop.svg    pieces/blue-bishop.svg
pieces/red-knight.svg    pieces/blue-knight.svg
pieces/red-boat.svg      pieces/blue-boat.svg
pieces/red-pawn.svg      pieces/blue-pawn.svg
pieces/yellow-*.svg      pieces/green-*.svg
```

Fehlende Dateien werden durch farbige Kreise ersetzt.
Gute Quelle: https://github.com/lichess-org/lila/tree/master/public/piece

---

## Empfohlene Trainingsparameter

| Ziel | Partien | Lambda | LR | Dauer (CPU) |
|------|---------|--------|----|-------------|
| Schnelltest | 200 | 0.5 | 0.01 | ~2 Min |
| Erste Ergebnisse | 2 000 | 0.7 | 0.001 | ~20 Min |
| Gute Qualität | 10 000 | 0.7 | 0.001 | ~90 Min |
| Hohe Qualität | 50 000 | 0.75 | 0.0005 | ~8 Std |

---

## Weiterentwicklung

- [ ] WebWorker für nicht-blockierende Engine-Suche im Browser
- [ ] Vollständiger PGN-Export (Zugliste)
- [ ] Zeitsteuerung (ms-Budget statt fixer Tiefe)
- [ ] Netz in die Alpha-Beta-Bewertung integrieren (hybride Eval-Funktion)
- [ ] Killer-Züge & History-Heuristik
- [ ] Multiplayer über WebRTC

## Lizenz

Der Code steht unter der **MIT-Lizenz** — siehe [LICENSE](LICENSE).

Die mitgelieferten Partiedaten nicht. Sie sind aus Chaturaji-Partien von
chess.com abgeleitet, und ihre Weitergabe richtet sich nach den
Nutzungsbedingungen von chess.com, nicht nach der MIT-Lizenz:

| Datei | Inhalt |
|---|---|
| `crates/wasm/www/opening_book.json` | aggregierte Statistiken je Stellung (Zughäufigkeit, mittlere Punkte und Platzierung) — keine Zuordnung zu einzelnen Partien oder Personen |
| `crates/wasm/www/games_index.json` | Partieübersicht mit **Benutzernamen, Ratings, Datum und Ergebnis** einzelner Spieler |

Beim zweiten Punkt geht es nicht nur um Urheberrecht: Benutzernamen und
Ratings sind personenbezogene Daten. Wer das Repository forkt oder die Dateien
weiterverbreitet, sollte das wissen.

Dasselbe gilt für die Trainingsdaten, die außerhalb des Repositories liegen
(`game_data/`, `game_data.tar.gz` im Release `nnue-state`). Die daraus
*gelernten* Netzgewichte sind davon unberührt — sie enthalten keine Partien,
sondern nur Parameter.
