//! Self-Play für NNUE-Training.
//!
//! Verwendet das NNUE-Netz direkt als Evaluierungsfunktion (Max^n mit TT).
//!
//! Beam-Suche: Interne Knoten sortieren Züge nach dem gelernten Zugmodell
//! (`chaturaji_engine::move_features`) und rekursieren nur in die besten
//! `beam_width` davon.
//! NNUE-Evals laufen ausschließlich an Blattknoten → ~beam_width^(depth-1)
//! Evals pro Wurzel-Zug statt B^(depth-1).
//!
//! Kosten (approx., B=30 Wurzelzüge):
//!   depth 1, beam 0 – ~30   Evals/Zug  (greedy, sehr schnell)
//!   depth 2, beam 0 – ~900  Evals/Zug
//!   depth 4, beam 3 – ~810  Evals/Zug  (30 × 3³)
//!   depth 4, beam 5 – ~3750 Evals/Zug  (30 × 5³)

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use rayon::prelude::*;
use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;
use chaturaji_core::notation::move_to_str;
use chaturaji_core::zobrist::{hash_board, ZobristKeys};
use chaturaji_engine::book::OpeningBook;
use chaturaji_engine::move_features::{fast_features, MoveFeatureContext, MoveModel};
use crate::mcts::{Fpu, Mcts, MctsConfig, RootChoice};
use crate::network::NnueNetwork;

pub struct SelfPlayConfig {
    pub epsilon_start:   f32,
    pub epsilon_end:     f32,
    pub epsilon_decay:   f32,
    /// Halbzüge, nach denen eine Partie abgebrochen wird.
    ///
    /// Der Wert ist keine Notbremse, sondern bestimmt mit, *welches Spiel* das
    /// Netz lernt. `Rules::is_game_over` ist allein „höchstens ein Spieler
    /// aktiv"; unter dieser Bedingung enden Selbstspiel-Partien praktisch nie
    /// von selbst — sie laufen bis zum Limit. Menschen dagegen geben auf oder
    /// spielen eine entschiedene Stellung nicht aus.
    ///
    /// Ausgezählt über 4000 echte Partien aus `game_data/`: Median 94
    /// Halbzüge, 90 % unter 152, nur gut 1 % erreichen 200. Ein Limit von 300
    /// füllte das Training daher zu einem großen Teil mit Stellungen, die in
    /// echten Partien nicht vorkommen.
    ///
    /// Der Abbruch verfälscht das Ziel nicht: `outcome::place_values` liest den
    /// Punktestand zum Abbruchzeitpunkt, und der ist auch bei unbeendeter
    /// Partie eine gültige Rangfolge.
    pub max_moves:       usize,
    pub engine_depth:    u8,
    /// Beam width at internal nodes (0 = unbegrenzt).
    /// Beim Beam werden Züge zuerst per 1-Ply-NNUE sortiert; nur die besten
    /// `beam_width` Züge werden rekursiv untersucht.
    /// Empfehlung: depth 1–2 → 0, depth 3 → 8, depth 4 → 6.
    pub beam_width:      usize,
    pub book_max_plies:  usize,
    pub book_min_count:  u32,
    /// Womit der Beam auswählt. Vorgabe: das gelernte Zugmodell.
    pub beam_order:      BeamOrder,
    /// Welche Suche die Partien erzeugt.
    pub generator:       Generator,
    /// Besuchsverteilung der Wurzel je Halbzug mitschreiben.
    ///
    /// Standardmäßig aus: sie kostet rund 25 KB je Partie und wird nur
    /// gebraucht, wenn auf sie trainiert werden soll. Ein gewöhnlicher
    /// Trainingslauf über 6.400 Partien würde damit 160 MB Artefakt erzeugen.
    pub record_visits:   bool,
}

/// Wie die Züge im Self-Play gesucht werden.
///
/// # Warum das zur Wahl steht
///
/// Die erzeugende Suche ist der **Lehrer**: das Netz lernt, sie zu destillieren.
/// Gemessen (dasselbe Netz, nur anders gesucht, je 576 Partien) schlägt MCTS
/// mit gelerntem Zug-Prior das bisherige Max^n mit Beam um **+0,54 bei gleicher
/// Rechenzeit** und um **+0,18 bei einem Viertel davon**. Auf der Kurve des
/// Beams — 0,051 Platzwert je Verdopplung des Aufwands — entspricht das rund
/// 1.600-facher Suche.
///
/// Ob ein besserer Lehrer auch einen besseren Fixpunkt ergibt, ist damit noch
/// nicht gezeigt: Startnetz und Kapazität waren ohne Wirkung, die Suchstärke
/// wirkte nur logarithmisch. Ein Sprung dieser Größe ist aber etwas anderes als
/// die Verdopplungen, die bisher geprüft wurden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Generator {
    /// Max^n mit Beam — der Weg bis 2026-09-14.
    Beam,
    /// Monte-Carlo-Baumsuche mit Zug-Prior. 1600 Simulationen sind
    /// kostengleich zu Tiefe 4 / Beam 6.
    Mcts { iterations: u32, c_puct: f32 },
}

impl Generator {
    /// Aus einem CLI-Wort. `mcts` nimmt die kostengleiche Vorgabe.
    pub fn from_str(s: &str, iterations: u32, c_puct: f32) -> Option<Self> {
        match s {
            "beam" | "maxn" => Some(Self::Beam),
            "mcts"          => Some(Self::Mcts { iterations, c_puct }),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self { Self::Beam => "beam", Self::Mcts { .. } => "mcts" }
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            epsilon_start:  0.3,
            epsilon_end:    0.05,
            epsilon_decay:  0.995,
            // 150 ≈ 90 %-Quantil echter Partien (152).
            max_moves:      150,
            engine_depth:   1,
            beam_width:     0,
            book_max_plies: 16,
            book_min_count: 2,
            beam_order:     BeamOrder::Model,
            generator:      Generator::Beam,
            record_visits:  false,
        }
    }
}

/// Wie viele Züge je Stellung aufgezeichnet werden.
///
/// Der Schwanz der Verteilung besteht aus Zügen mit ein bis zwei Besuchen; bei
/// 800 Simulationen auf ~30 Züge trägt er nichts zum Ziel bei und verdoppelte
/// die Dateigröße. Sechzehn decken alles ab, was nennenswert besucht wurde.
pub const VISIT_TOP_K: usize = 16;

pub struct Step {
    /// Die vollständige Stellung, nicht nur die Bitboards: das Netz braucht
    /// auch Punktestand, Zugrecht und Spielphase (siehe `features::dense_features`).
    pub board: Board,
    pub value: [f32; 4],
    /// Bewertung der Stellung *nach der Suche* — der Scorevektor des besten
    /// Zuges aus `nnue_best_move_scored`.
    ///
    /// Das ist eine bessere Schätzung als `value`: dort steht die rohe Ausgabe
    /// des Netzes für diese Stellung, hier das Ergebnis von `engine_depth`
    /// Halbzügen Vorausschau mit demselben Netz an den Blättern. Die Suche
    /// korrigiert das Netz — genau das macht sie ja — und diese Korrektur ist
    /// ein Lernsignal, das im TD-Training bisher weggeworfen wurde.
    ///
    /// `None`, wenn keine Suche lief: bei einem Buchzug und bei einem
    /// ε-Zufallszug. Beides wäre nachträglich nur mit einer zweiten Suche zu
    /// füllen, und die kostet so viel wie die erste.
    pub search_value: Option<[f32; 4]>,
    /// Besuchszahl je Zug an der Wurzel, absteigend und auf die wichtigsten
    /// gekürzt.
    ///
    /// Das ist das Trainingsziel für den Zug-Prior: die Suche verteilt ihre
    /// Aufmerksamkeit besser, als der Prior sie vorhergesagt hat — sonst
    /// brächte sie nichts —, und genau diese Differenz ist das Lernsignal.
    ///
    /// Leer, wenn nicht mit MCTS gesucht wurde (Buchzug, ε-Zufallszug, oder
    /// der Beam als Erzeuger). Gekürzt, weil die Verteilung einen langen
    /// Schwanz aus Zügen mit ein bis zwei Besuchen hat, der nichts trägt und
    /// die Dateien verdoppelte.
    pub visits: Vec<(u8, u8, u32)>,
}

pub struct GameResult {
    pub steps:       Vec<Step>,
    pub final_board: Board,
    pub move_log:    Vec<String>,
    pub winner:      Option<Color>,
}

// ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

/// Das gelernte Zugmodell für die Beam-Auswahl.
///
/// Bis 2026-09-12 stand hier eine Handheuristik: Schlagwert plus 20 für eine
/// Umwandlung. Gemessen an 86.221 Zugentscheidungen von Spielern ab 2400, die
/// nicht zum Lernen benutzt wurden — wie oft überlebt der Zug, den ein starker
/// Spieler wählt, den Beam der besten sechs?
///
/// | | Top-1 | im Beam-6 |
/// |---|---|---|
/// | zufällige Reihenfolge | 8,0 % | 47,5 % |
/// | die alte Heuristik | 17,0 % | 54,6 % |
/// | **das Modell** | **25,6 %** | **74,3 %** |
///
/// Die alte Heuristik lag also nur sieben Punkte über dem Zufall: fast die
/// Hälfte der Züge, die ein starker Spieler wählen würde, fiel aus dem Beam
/// und wurde nie angesehen.
///
/// Der Beam ist die Stelle, an der das zählt — hier werden Züge **verworfen**.
/// Im Alpha-Beta der Engine (`chaturaji_engine::ordering`) ist MVV-LVA
/// weiterhin besser, weil es dort um frühe Schnitte geht und nicht um
/// Menschenähnlichkeit; dort kostete dasselbe Modell 12 % mehr Knoten.
static MOVE_MODEL: std::sync::OnceLock<MoveModel> = std::sync::OnceLock::new();

fn move_model() -> &'static MoveModel {
    MOVE_MODEL.get_or_init(MoveModel::default)
}

/// Womit der Beam seine Züge auswählt.
///
/// Die Variante ist ein Parameter und keine feste Entscheidung, damit die
/// Arena beide Seiten unterschiedlich spielen lassen kann. Ohne das ließe sich
/// eine Sortierung nicht gegen die andere messen: die Arena baut beide Seiten
/// aus demselben Quellstand, und ein Vergleich zweier *Netze* sagt über die
/// Sortierung nichts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BeamOrder {
    /// Das gelernte Zugmodell aus echten Partien.
    #[default]
    Model,
    /// Die Handheuristik von vor 2026-09-12: Schlagwert plus 20 für eine
    /// Umwandlung. Nur noch zum Vergleichen da.
    Legacy,
}

impl BeamOrder {
    /// Aus einem CLI-Wort. Unbekanntes ergibt `None`, damit der Aufrufer
    /// meckern kann, statt still die Vorgabe zu nehmen.
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "model"  | "modell" => Some(Self::Model),
            "legacy" | "alt"    => Some(Self::Legacy),
            _ => None,
        }
    }
}

/// Die alte Handheuristik: Schlagwert plus Umwandlungsbonus.
#[inline]
fn legacy_move_priority(mv: Move) -> i32 {
    let capture = match mv.captured.map(|p| p.kind) {
        Some(PieceKind::King)   => 100,
        Some(PieceKind::Boat)   => 50,
        Some(PieceKind::Knight) => 30,
        Some(PieceKind::Bishop) => 30,
        Some(PieceKind::Pawn)   => 10,
        None                    => 0,
    };
    capture + if mv.promoted { 20 } else { 0 }
}

/// Kürzt `moves` auf die besten `beam_width` nach der gewählten Sortierung.
fn apply_beam(board: &Board, moves: &mut Vec<Move>, beam_width: usize, order: BeamOrder) {
    if beam_width == 0 || moves.len() <= beam_width { return; }
    match order {
        BeamOrder::Legacy => {
            moves.sort_by_key(|&mv| std::cmp::Reverse(legacy_move_priority(mv)));
            moves.truncate(beam_width);
        }
        BeamOrder::Model => {
            // Der Kontext (vier Angriffskarten) einmal je Stellung — nicht je Zug.
            let model = move_model();
            let ctx   = MoveFeatureContext::new(board);
            let mut bewertet: Vec<(Move, f32)> = moves.iter()
                .map(|mv| (*mv, model.score_features(&fast_features(board, mv, &ctx))))
                .collect();
            bewertet.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            *moves = bewertet.into_iter().take(beam_width).map(|(mv, _)| mv).collect();
        }
    }
}

// ─── NNUE Max^n mit Transpositionstabelle ────────────────────────────────────

/// Rekursiver Max^n mit NNUE-Blattbewertung und optionalem Beam.
///
/// `beam_width > 0`: Interne Knoten sortieren Züge nach dem gelernten
/// Zugmodell (kein NNUE-Aufruf) und rekursieren nur in die besten `beam_width`.
/// NNUE-Evals laufen ausschließlich an Blattknoten (depth == 0).
///
/// Die TT speichert (Tiefe, Scorevektor); ein Eintrag wird nur verwendet,
/// wenn die gespeicherte Tiefe ≥ der angefragten Tiefe ist.
pub fn nnue_maxn(
    net:        &NnueNetwork,
    board:      &Board,
    depth:      u8,
    beam_width: usize,
    order:      BeamOrder,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
    keys:       &ZobristKeys,
) -> [f32; 4] {
    if Rules::is_game_over(board) || depth == 0 {
        return net.forward(board);
    }

    let hash = hash_board(board, keys);
    if let Some(&(d, scores)) = tt.get(&hash) {
        if d >= depth { return scores; }
    }

    let mut all_moves = Rules::legal_moves(board);
    if all_moves.is_empty() {
        return net.forward(board);
    }

    let mover_idx = board.to_move.idx();

    apply_beam(board, &mut all_moves, beam_width, order);
    let moves = all_moves;

    let mut best = [f32::NEG_INFINITY; 4];

    for mv in moves {
        let child  = Rules::apply_with_effects(board, mv);
        let scores = nnue_maxn(net, &child, depth - 1, beam_width, order, tt, keys);
        if scores[mover_idx] > best[mover_idx] {
            best = scores;
        }
    }

    tt.insert(hash, (depth, best));
    best
}

/// Bewertet alle legalen Züge mit NNUE Max^n und gibt den besten zurück.
/// Die Wurzel betrachtet immer alle legalen Züge (kein Beam auf Wurzelebene).
/// Die TT wird vor jedem Aufruf geleert (korrektes Tiefenhandling über Züge hinweg).
pub fn nnue_best_move(
    net:        &NnueNetwork,
    board:      &Board,
    moves:      &[Move],
    depth:      u8,
    beam_width: usize,
    order:      BeamOrder,
    keys:       &ZobristKeys,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
) -> Move {
    nnue_best_move_scored(net, board, moves, depth, beam_width, order, keys, tt).0
}

/// Wie [`nnue_best_move`], gibt aber zusätzlich den **vollen Scorevektor** des
/// gewählten Zuges zurück — die Bewertung der Stellung nach der Suche.
///
/// Der Vektor ist der Rückgabewert von `nnue_maxn` für das Kind des besten
/// Zuges, also die Einschätzung aller vier Spieler nach `depth` Halbzügen
/// Vorausschau. Das Generationentraining benutzt ihn als Teil des Zielwerts
/// (siehe [`crate::gen_train`]); `nnue_best_move` wirft ihn weg.
pub fn nnue_best_move_scored(
    net:        &NnueNetwork,
    board:      &Board,
    moves:      &[Move],
    depth:      u8,
    beam_width: usize,
    order:      BeamOrder,
    keys:       &ZobristKeys,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
) -> (Move, [f32; 4]) {
    let mover_idx = board.to_move.idx();
    let d1 = depth.saturating_sub(1);

    moves.iter().copied()
        .map(|mv| {
            let child  = Rules::apply_with_effects(board, mv);
            let scores = nnue_maxn(net, &child, d1, beam_width, order, tt, keys);
            (mv, scores)
        })
        .max_by(|a, b| {
            a.1[mover_idx].partial_cmp(&b.1[mover_idx]).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or((moves[0], net.forward(board)))
}

/// Iterative Vertiefung mit Zeitbudget. Gibt Zug, Scorevektor und die zuletzt
/// **vollständig** gerechnete Tiefe zurück.
///
/// # Warum es das braucht
///
/// Bei fester Tiefe misst die Arena gleiche Vorausschau, nicht gleiche
/// Rechenzeit. Zwei Zugsortierungen unterscheiden sich aber gerade darin, was
/// ein Knoten kostet: die gelernte Sortierung baut vier Angriffskarten je
/// Stellung und ist damit rund 14 % langsamer. Bei gleicher Tiefe zahlt sie
/// diesen Preis, ohne ihn anrechnen zu müssen — bei gleicher Zeit sucht die
/// billigere Sortierung dafür tiefer. Erst der zweite Vergleich sagt, ob sich
/// der Aufwand lohnt.
///
/// # Wie abgebrochen wird
///
/// Zwischen den Iterationen, nicht mittendrin: `nnue_maxn` hat keine
/// Abbruchprüfung, und eine halb gerechnete Tiefe wäre auch kein brauchbares
/// Ergebnis. Ob die nächste Tiefe noch hineinpasst, wird aus dem gemessenen
/// Wachstumsfaktor der bisherigen Iterationen geschätzt. Lieber eine Iteration
/// zu wenig als das Budget zu reißen — sonst bekäme die langsamere Sortierung
/// unbemerkt mehr Zeit, und genau das soll die Messung ja ausschließen.
pub fn nnue_best_move_timed(
    net:        &NnueNetwork,
    board:      &Board,
    moves:      &[Move],
    budget_ms:  u64,
    max_depth:  u8,
    beam_width: usize,
    order:      BeamOrder,
    keys:       &ZobristKeys,
    tt:         &mut HashMap<u64, (u8, [f32; 4])>,
) -> (Move, [f32; 4], u8) {
    let start = std::time::Instant::now();
    let budget = std::time::Duration::from_millis(budget_ms);

    // Tiefe 1 wird immer gerechnet — ohne sie gäbe es keinen Zug.
    let (mut best_mv, mut best_scores) =
        nnue_best_move_scored(net, board, moves, 1, beam_width, order, keys, tt);
    let mut erreicht = 1u8;
    let mut letzte   = start.elapsed();
    // Erster Schätzwert für den Sprung auf die nächste Tiefe. Der Beam begrenzt
    // die Verzweigung auf `beam_width`; ohne Beam ist sie die volle Zugzahl.
    let mut faktor = if beam_width > 0 { beam_width as f64 } else { moves.len().max(2) as f64 };

    for depth in 2..=max_depth {
        let geschaetzt = letzte.mul_f64(faktor);
        if start.elapsed() + geschaetzt > budget { break; }

        let t0 = std::time::Instant::now();
        let (mv, scores) =
            nnue_best_move_scored(net, board, moves, depth, beam_width, order, keys, tt);
        let dauer = t0.elapsed();

        best_mv     = mv;
        best_scores = scores;
        erreicht    = depth;

        // Gemessener Faktor schlägt die Annahme, sobald es einen gibt.
        if letzte.as_secs_f64() > 1e-6 {
            faktor = (dauer.as_secs_f64() / letzte.as_secs_f64()).max(1.5);
        }
        letzte = dauer;
    }

    (best_mv, best_scores, erreicht)
}

// ─── play_game ────────────────────────────────────────────────────────────────

pub fn play_game(
    net:     &NnueNetwork,
    cfg:     &SelfPlayConfig,
    epsilon: f32,
    rng:     &mut impl Rng,
    book:    Option<&OpeningBook>,
    keys:    &ZobristKeys,
) -> GameResult {
    let mut board    = Board::default();
    let mut steps    = Vec::with_capacity(cfg.max_moves);
    let mut move_log = Vec::with_capacity(cfg.max_moves);
    let mut tt: HashMap<u64, (u8, [f32; 4])> = HashMap::new();
    // Baum und Zugmodell einmal je Partie. Der Baum wird je Zug geleert, seine
    // Allokationen bleiben erhalten.
    let mut baum   = Mcts::new();
    let modell     = MoveModel::default();

    for ply in 0..cfg.max_moves {
        if Rules::is_game_over(&board) { break; }

        let moves = Rules::legal_moves(&board);
        if moves.is_empty() { break; }

        let value = net.forward(&board);

        let book_move = book
            .filter(|_| ply < cfg.book_max_plies)
            .and_then(|b| b.entries(&board, keys, cfg.book_min_count))
            .and_then(|entries| sample_book_move(&entries, &moves, rng));

        let mut search_value = None;
        let mut besuche: Vec<(u8, u8, u32)> = Vec::new();
        let chosen = if let Some(mv) = book_move {
            mv
        } else if rng.gen::<f32>() < epsilon {
            moves[rng.gen_range(0..moves.len())]
        } else {
            match cfg.generator {
                Generator::Beam => {
                    tt.clear();
                    let (mv, scores) = nnue_best_move_scored(
                        net, &board, &moves, cfg.engine_depth, cfg.beam_width,
                        cfg.beam_order, keys, &mut tt,
                    );
                    search_value = Some(scores);
                    mv
                }
                Generator::Mcts { iterations, c_puct } => {
                    let r = baum.search(&|b: &Board| net.forward(b), &modell, &board,
                                        &MctsConfig { iterations, c_puct, root: RootChoice::Visits,
                                                      fpu: Fpu::Null, reuse: false });
                    match r {
                        Some(r) => {
                            // Die Wurzelbewertung von MCTS ist der Mittelwert
                            // über alle Simulationen und damit eine bessere
                            // Schätzung als der Scorevektor eines einzelnen
                            // Max^n-Pfades.
                            search_value = Some(r.value);
                            if cfg.record_visits {
                                besuche = r.visits.iter().take(VISIT_TOP_K)
                                    .map(|(m, n)| (m.from, m.to, *n))
                                    .collect();
                            }
                            r.best
                        }
                        None => moves[0],
                    }
                }
            }
        };

        steps.push(Step { board: board.clone(), value, search_value, visits: besuche });
        move_log.push(move_to_str(&chosen));
        board = Rules::apply_with_effects(&board, chosen);
    }

    let winner = Rules::winner(&board);
    GameResult { steps, final_board: board, move_log, winner }
}

// ─── Paralleles Self-Play ─────────────────────────────────────────────────────

/// Eine zu spielende Partie: laufende Nummer, RNG-Seed und ε.
///
/// Der Seed hängt allein an der globalen Partienummer, nicht an der Reihenfolge
/// der Abarbeitung. Damit liefert derselbe Lauf dasselbe Ergebnis, egal auf wie
/// vielen Kernen oder in wie vielen Shards er läuft — ohne das wäre ein
/// verteilter Lauf nicht reproduzierbar und ein Fehler nicht nachstellbar.
#[derive(Clone, Copy)]
pub struct GameJob {
    pub index:   u64,
    pub seed:    u64,
    pub epsilon: f32,
}

/// Ableitung des Partie-Seeds aus Lauf-Seed und Partienummer.
pub fn game_seed(run_seed: u64, index: u64) -> u64 {
    // SplitMix64-Finalizer: streut benachbarte Indizes weit auseinander, damit
    // Partie 7 und Partie 8 nicht mit verwandten Zufallsfolgen starten.
    let mut z = run_seed
        .wrapping_add(index.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Spielt einen Stapel Partien parallel gegen dasselbe, eingefrorene Netz.
///
/// Das Netz wird nur gelesen (`forward` nimmt `&self`), deshalb braucht es
/// keinen Lock. Die Rückgabe ist nach `GameJob::index` sortiert, damit die
/// anschließenden TD-Updates in fester Reihenfolge laufen.
pub fn play_batch(
    net:  &NnueNetwork,
    cfg:  &SelfPlayConfig,
    jobs: &[GameJob],
    book: Option<&OpeningBook>,
    keys: &ZobristKeys,
) -> Vec<(u64, GameResult)> {
    let mut out: Vec<(u64, GameResult)> = jobs
        .par_iter()
        .map(|job| {
            let mut rng = SmallRng::seed_from_u64(job.seed);
            (job.index, play_game(net, cfg, job.epsilon, &mut rng, book, keys))
        })
        .collect();
    out.sort_by_key(|(idx, _)| *idx);
    out
}

fn sample_book_move(
    entries: &[(u8, u8, u32)],
    legal:   &[Move],
    rng:     &mut impl Rng,
) -> Option<Move> {
    let total: u64 = entries.iter().map(|(_, _, c)| *c as u64).sum();
    if total == 0 { return None; }
    let mut pick = rng.gen_range(0..total);
    for (from, to, count) in entries {
        let c = *count as u64;
        if pick < c {
            return legal.iter().find(|m| m.from == *from && m.to == *to).copied();
        }
        pick -= c;
    }
    None
}

/// Endstand → Zielvektor für das Training: die Platzwertung des Endergebnisses.
///
/// Dieselbe Kodierung wie im PGN-/JSON-Import (`outcome::place_values`), damit
/// Self-Play und Supervised Learning dasselbe lernen.
pub fn final_targets(board: &Board) -> [f32; 4] {
    // `Rules::final_scores` statt `board.scores`: bleibt genau ein Spieler
    // übrig, gehören ihm die 3 Punkte je nie geschlagenem König. Im Self-Play
    // ändert das nichts — dort scheidet man nur durch den Verlust des Königs
    // aus, es bleibt also keiner stehen. Beide Wege sollen aber dieselbe
    // Rechnung benutzen, damit Self-Play und echte Partien dasselbe Ziel
    // lernen.
    crate::outcome::place_values(Rules::final_scores(board))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn play_short_game_produces_steps() {
        let net = NnueNetwork::new(0.001, 0.9);
        let cfg = SelfPlayConfig { max_moves: 20, ..Default::default() };
        let mut rng = rand::thread_rng();
        let keys = ZobristKeys::new();
        let result = play_game(&net, &cfg, 1.0, &mut rng, None, &keys);
        assert!(!result.steps.is_empty());
        assert!(!result.move_log.is_empty());
    }

    #[test]
    fn nnue_greedy_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 1, 0, BeamOrder::Model, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn nnue_depth2_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 2, 0, BeamOrder::Model, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn nnue_depth4_beam_picks_a_move() {
        let net   = NnueNetwork::new(0.001, 0.9);
        let board = Board::default();
        let moves = Rules::legal_moves(&board);
        let keys  = ZobristKeys::new();
        let mut tt = HashMap::new();
        let mv = nnue_best_move(&net, &board, &moves, 4, 6, BeamOrder::Model, &keys, &mut tt);
        assert!(moves.contains(&mv));
    }

    #[test]
    fn final_targets_in_range() {
        for v in final_targets(&Board::default()) {
            assert!((-1.0..=1.0).contains(&v));
        }
    }
}
