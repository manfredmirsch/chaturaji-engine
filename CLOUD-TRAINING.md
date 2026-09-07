# NNUE-Training auf GitHub Actions

Das Training läuft nicht mehr als ein langer Prozess auf einer Maschine, sondern
rundenweise über viele Runner. Dieses Dokument sagt, warum, wie man es startet
und wo die Grenzen liegen.

## Warum das nötig war

`td::run` ist strikt sequentiell: Partie spielen → Gewichte aktualisieren →
nächste Partie mit den neuen Gewichten. Das lässt sich nicht verteilen, und der
Trainer war zusätzlich single-threaded — auf einem 4-vCPU-Runner hätten drei
Kerne leergelaufen.

Ein GitHub-Runner ist **pro Kern nicht schneller** als ein Notebook. Der Gewinn
kommt allein aus der Parallelität: 16 Runner × 4 vCPU, kostenlos, weil das
Repository öffentlich ist. Dafür musste das Verfahren umgebaut werden.

## Das Rundenverfahren

```
Runde r:
  16 Shards spielen parallel Partien gegen die eingefrorenen Gewichte w_r
  und legen nur die Zugfolgen ab (wenige hundert Byte je Partie).
        ↓
  Ein Lern-Job spielt alle Partien der Runde in der Reihenfolge ihrer
  globalen Partienummer nach und wendet dieselben TD(λ)-Updates an
  wie der lokale Trainer.                                    →  w_{r+1}
```

Der einzige inhaltliche Unterschied zum lokalen Lauf: die Züge einer Runde
stammen von einem Netz, das bis zu eine Runde alt ist. Lokal war das Netz
innerhalb einer Partie eingefroren (die Updates liefen erst nach Partieende) —
die Staleness wächst also von „eine Partie" auf „eine Runde". Das ist die
übliche Bauform für verteiltes Self-Play, aber es ist eine Abweichung, und wer
die Rundengröße hochdreht, dreht sie mit hoch.

Alles, was den Trainingsverlauf bestimmt, hängt jetzt an der **globalen
Partienummer** statt am Prozessstart:

| Größe | vorher | jetzt |
|---|---|---|
| ε | begann bei jedem Start wieder bei 0,3 | `epsilon_at(games_done)` |
| Lernrate | wurde beim Start auf `--lr` zurückgesetzt | `lr_at(games_done)` |
| RNG | fester Seed 42 → jeder Neustart spielte dieselben Partien | Seed je Partienummer |
| Adam-Momente | `#[serde(skip)]`, bei jedem Start 0 | `opt_state.bin` |

Die ersten drei waren bei einem 13-Stunden-Prozess folgenlos und bei einem Lauf
über 30 Jobs nicht mehr. Der vierte Punkt kostete bei jedem Neustart einen um
etwa Faktor 3 zu großen ersten Adam-Schritt.

Zwei Tests halten die neuen Zeitpläne an die alte Schleife gekoppelt
(`dist::tests::epsilon_schedule_matches_loop`, `lr_schedule_matches_loop`), ein
dritter prüft, dass das Nachspielen exakt die Stellungen des Self-Plays
rekonstruiert.

## Einmalige Einrichtung

Der Trainingszustand liegt in einem Release, nicht im Git-Verlauf — sonst wüchse
das Repository je Runde um 7 MB.

1. **Workflows auf den Standard-Branch bringen.** `workflow_dispatch` zeigt
   Workflows nur an, wenn sie auf `main` liegen.

2. **Release `nnue-state` anlegen** und diese drei Dateien anhängen
   (liegen bereits fertig in `~/chaturaji/cloud-state/`):

   | Datei | Inhalt |
   |---|---|
   | `weights.json` | letzter Checkpoint aus `nnue.db` (3.181.677 Schritte) |
   | `progress.json` | `games_done: 15037`, `run_seed: 4242` |
   | `opening_book.json` | das Buch — liegt außerhalb des Repos und muss mit |

   Über die Weboberfläche: *Releases → Draft a new release → Tag `nnue-state`
   → Dateien anhängen → Publish*.

   Neu erzeugen ließen sie sich mit:
   ```bash
   train_nnue --init-state --db nnue.db \
     --weights-out weights.json --progress progress.json \
     --games-done 15037 --seed 4242
   ```

   `games_done: 15037` ist die Partienzahl des bisherigen Laufs. Sie hält ε auf
   dem Boden (0,05) und setzt den Lernraten-Zerfall fort: bei `--lr 0.0013`
   ergibt das die zuletzt genutzten ≈ 0,00061.

## Starten

*Actions → NNUE-Training → Run workflow.* Voreinstellungen:

| Eingabe | Standard | Bedeutung |
|---|---|---|
| `rounds` | 4 | Runden in diesem Lauf (max. 8; danach erneut starten) |
| `games` | 6400 | Partien je Runde, über alle Shards zusammen |
| `shards` | 16 | parallele Erzeuger (Kontingent: 20 gleichzeitige Jobs) |
| `depth` / `beam_width` | 4 / 6 | Suchtiefe im Self-Play |
| `lr` | 0,0013 | Basis-Lernrate *vor* dem Zerfall, nicht die effektive |
| `max_seconds` | 2700 | Notbremse je Erzeuger, kein Sollwert |

Gemessen (lokal, 4 Threads, `depth 4 --beam-width 6`): **20 Partien/min**,
∅ 81,5 Halbzüge. Ein Shard braucht für seine 400 Partien also gut 25 Minuten,
der Lern-Job für 6400 Partien etwa 7 (er schafft rund 16 Partien/s). Ein Lauf
mit 4 Runden liegt bei ~3,5 Stunden und 25.600 Partien — zum Vergleich: der
bisherige lokale Lauf brauchte 13,5 Stunden für 10.000 Partien bei Tiefe 1.

`max_seconds` greift nur, wenn ein Runner deutlich langsamer ist als erwartet.
Dann bricht der Shard sauber ab und lädt hoch, was er hat; die Runde lernt
entsprechend weniger Partien und der Fortschrittszähler zählt nur die
tatsächlich gelernten. Ohne das würde der 6-Stunden-Deckel von GitHub den Job
hart abschießen — und ein hart abgeschossener Job lädt kein Artefakt hoch.

## Ergebnis abholen

Nach jeder Runde liegen die neuen Gewichte im Release `nnue-state`
(`weights.json`) und zusätzlich als Artefakt `nnue-weights-round-N`. Für die
Arena oder das WASM-Frontend einfach herunterladen und wie bisher verwenden.

Der Fortschritt steht in der Zusammenfassung jedes Lern-Jobs (∅ Loss, ∅ Züge,
Schritte, Lernrate) und in `progress.json`.

## Supervised Pre-Training

*Actions → NNUE-Pre-Training → Run workflow.* Ein einzelner Job, keine Shards:
SGD geht die Stellungen der Reihe nach durch, da ist nichts zu parallelisieren.

Gemessen: ~1.900 Stellungen/s, also gut **neun Minuten** für den ganzen
Datensatz, bei 35 MB Speicherbedarf. Aus 1.000 Dateien werden 1.000 Partien mit
116.160 Stellungen.

Die Regelimplementierung ist an allen 11.558 Partien geprüft: alle sind
nachspielbar, und bei 11.555 davon stimmt der nachgespielte Endstand exakt mit
`points1..4` von chess.com überein. Die drei übrigen wurden mit einer anderen
Einstellung gespielt — ihr Systemchat sagt „Checkmates/kings = +10 points"
statt der sonst überall geltenden +3. Nachprüfen lässt sich das jederzeit mit

```bash
cargo run --release -p chaturaji-nnue --example diagnose_pgn -- ~/chaturaji/game_data
```

Dafür muss der Datensatz einmalig ans Release `nnue-state` — siehe
[Training von vorn beginnen](#training-von-vorn-beginnen), Schritt 1.

Das Ergebnis überschreibt `weights.json` **nicht**, sondern liegt als
`weights-pretrained.json` daneben. Ein Pre-Training zieht das Netz auf eine
andere Zielverteilung — ob das besser ist, entscheidet die Arena, nicht der
Workflow. Übernehmen ist ein bewusster zweiter Schritt.

**Das TD-Netz sagt echte Ausgänge schlechter vorher als ein untrainiertes.**
Auf denselben 758 Partien gemessen:

| Netz | ∅ Loss |
|---|---|
| frisch initialisiert | 0,19 – 0,28 |
| TD-Netz, 3.181.677 Schritte | 1,31 – 1,44 |
| Referenz: konstante Ausgabe 0 | ≈ 0,55 |

Schlechter als konstant Null ist keine bloße Verteilungsverschiebung. Solange
das nicht geklärt ist, ist ein Pre-Training auf dem TD-Netz ein Schuss ins
Dunkle — deshalb die Option `start_from: frisch`.

## Training von vorn beginnen

Nötig wird das, wenn sich die Punktelogik geändert hat: der Punktestand geht
über `dense_features` als Eingabe ins Netz und über `place_values` ins
Trainingsziel. Ein Netz, das mit falschen Punkten trainiert wurde, hat etwas
anderes gelernt als das Spiel, das jetzt implementiert ist.

Drei Schritte, keiner davon Handarbeit an Dateien:

1. **Einmalig** den Datensatz ans Release `nnue-state` hängen:

   ```bash
   python3 scripts/pack-game-data.py ~/chaturaji/game_data ~/chaturaji/game_data.tar.xz
   ```

   Das Skript wirft alles weg, was das Training nie liest — Avatare, Chat,
   IP-Hashes, Ratings — und behält Zugfolge und Endstand. Nötig ist das nicht
   aus Sparsamkeit: die Weboberfläche von GitHub nimmt nur Dateien bis 25 MB,
   und das vollständige Archiv liegt mit 32 MB darüber.

   | Variante | Größe |
   |---|---|
   | vollständig, gzip | 32 MB — Upload wird abgelehnt |
   | reduziert, gzip | 21 MB |
   | reduziert, xz | **13 MB** |

   Geprüft: mit den reduzierten Dateien liefert `diagnose_pgn` dieselben
   Zahlen wie mit den vollständigen.

2. *Actions → NNUE-Pre-Training → Run workflow* mit

   | Eingabe | Wert |
   |---|---|
   | `start_from` | `frisch` |
   | `promote` | ✔ |
   | `run_seed` | eine neue Zahl |
   | `epochs` | 1 (oder mehr) |

   Das trainiert ein frisch initialisiertes Netz auf den echten Partien und
   übernimmt das Ergebnis anschließend als `weights.json`. Der Fortschritt wird
   auf `games_done: 0` zurückgesetzt — ε beginnt wieder bei 0,3 und die
   Lernrate beim vollen `--lr`; ohne das liefe ein frisches Netz mit dem
   Zeitplan eines ausgereiften weiter.

   Die bisherigen Gewichte gehen nicht verloren: sie werden vorher als
   `weights-legacy.json` ins Release gesichert.

3. *Actions → NNUE-Training → Run workflow* wie gewohnt. Bei `lr` jetzt den
   vollen Startwert nehmen (`0.001`), nicht die 0,0013 aus der Fortsetzung des
   alten Laufs — die waren nur dazu da, den bereits gelaufenen
   Lernraten-Zerfall auszugleichen.

Das Eröffnungsbuch muss nicht neu gebaut werden. Es liest nur die ersten 12–16
Halbzüge, und in keiner der 11.558 Partien gibt es dort eine Aufgabe oder
Zeitüberschreitung — der Importer-Fehler, der ein Viertel der Partien verwarf,
hat es nie erreicht.

## Grenzen, die man kennen sollte

- **Kein Ersatz für eine schnelle Maschine.** Pro Kern ist ein Runner eher
  langsamer. Wer einen einzelnen, langen, streng sequentiellen Lauf will, ist
  auf einer dedizierten vCPU besser bedient.
- **Rundengröße = Staleness.** 6400 Partien je Update-Batch ist deutlich mehr
  als die bisherigen „nach jeder Partie". Wenn der Loss sich verschlechtert,
  ist `games` der erste Regler — kleinere Runden sind treuer zum lokalen
  Verfahren, kosten aber je Runde ~8 Minuten Rüstzeit.
- **Ein Lauf zur Zeit.** `concurrency: nnue-training` stellt zweite Läufe an,
  statt sie parallel in dieselben Release-Assets schreiben zu lassen.
- **Erzeuger und Lerner müssen denselben Commit fahren.** Sonst laufen
  Nachspielen und Erzeugen auseinander; der Lern-Job prüft das je Partie über
  den Endpunktestand und überspringt Abweichungen mit einer Warnung.
- **GitHub Actions ist für die Software des Repositories da.** Die Gewichte
  gehören zu dieser Engine, das ist gedeckt; Dauerlast über Wochen wäre eine
  andere Frage.

## Lokal weiterarbeiten

Der bisherige Weg bleibt unverändert:

```bash
cargo run --release -p chaturaji-nnue --bin train_nnue -- --games 5000 --depth 1
```

Zusätzlich profitiert er jetzt vom `[profile.release]` im Workspace-Root
(`lto = "fat"`, `codegen-units = 1`). Das `[profile.release]` in
`crates/wasm/Cargo.toml` war wirkungslos — Cargo wertet Profile nur aus dem
Root-Manifest aus.

Ein verteilter Lauf lässt sich auch lokal nachstellen:

```bash
train_nnue --init-state --db nnue.db --weights-out w.json --progress p.json
train_nnue --generate games/s0.jsonl --weights w.json --progress p.json \
           --shards 2 --shard 0 --games 8 --depth 4 --beam-width 6
train_nnue --generate games/s1.jsonl --weights w.json --progress p.json \
           --shards 2 --shard 1 --games 8 --depth 4 --beam-width 6
train_nnue --learn games --weights w.json --progress p.json --lr 0.0013
```
