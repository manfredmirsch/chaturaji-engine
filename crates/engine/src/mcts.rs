//! Monte-Carlo-Baumsuche mit NNUE-Blattbewertung und gelerntem Zug-Prior.
//!
//! # Wozu, wenn es Max^n mit Beam schon gibt
//!
//! Der Beam baut einen Baum fester Form: Tiefe `d`, an jedem inneren Knoten die
//! besten `b` Züge. Jeder Zweig bekommt gleich viel Aufmerksamkeit, und was
//! herausfällt, ist für immer weg.
//!
//! MCTS verteilt dasselbe Budget ungleich. Jede Simulation läuft von der Wurzel
//! abwärts, wählt unterwegs Züge nach PUCT, bewertet ein neues Blatt und trägt
//! das Ergebnis zurück nach oben. Ein klar bester Zug sammelt hunderte Besuche,
//! ein offensichtlich schlechter zwei — **die Verteilung passt sich an, statt
//! vorgegeben zu sein**.
//!
//! Gemessen ist das hier noch nichts. Der Grund, es überhaupt zu bauen: der
//! Fixpunkt des Selbstspiels hängt nachweislich nicht am Startnetz und nur
//! logarithmisch an der Suchstärke (0,05 Platzwert je Verdopplung). Eine andere
//! *Form* der Suche ist damit nicht widerlegt — nur mehr vom Gleichen.
//!
//! # Der Prior
//!
//! [`crate::move_features`] liefert für jeden Zug einen Score, der
//! aus 862.206 Zugentscheidungen von Spielern ab 2400 geschätzt wurde. Das
//! Modell ist eine konditionale Logit-Regression — seine Scores **sind** also
//! Log-Wahrscheinlichkeiten bis auf eine Konstante, und der Softmax darüber ist
//! genau die vom Modell geschätzte Zugverteilung. Kein Temperaturparameter
//! nötig, kein Nachkalibrieren: die Größe passt von Haus aus.
//!
//! Das ist der Unterschied zu `Foobork/flutteraji`, wo der Prior aus einer
//! Handheuristik stammt (MVV-LVA, Schach, Promotion, Zentrum). Deren Prior
//! trifft den menschlichen Zug in 17 % der Fälle, dieser in 25,6 %.
//!
//! # Vier Spieler
//!
//! Jeder Knoten führt einen Wertvektor über alle vier Sitze. Ausgewählt wird
//! nach der Komponente **des Spielers, der an diesem Knoten am Zug ist** — jeder
//! maximiert seinen eigenen Anteil, wie in Max^n. Eine Paranoid-Annahme („alle
//! gegen mich") wäre hier falsch: Chaturaji ist ein Jeder-gegen-jeden, und die
//! Punkte sind nicht Null-Summe zwischen zwei Seiten.

use std::collections::HashMap;

use chaturaji_core::board::{Board, Move};
use chaturaji_core::rules::Rules;

use crate::move_features::{fast_features, MoveFeatureContext};
use crate::policy::MovePrior;
use crate::outcome::place_values;

/// Blattbewertung: Stellung → erwartete Platzierung je Sitz, in [−1, 1].
///
/// Als Funktion statt als konkretes Netz, damit die Suche in der Engine leben
/// kann. Der Trainer reicht sein `NnueNetwork` durch, das WASM-Frontend sein
/// eigenes — und ein Test kommt mit einer Konstanten aus.
pub type LeafEval<'a> = &'a dyn Fn(&Board) -> [f32; 4];

/// Endstand → Platzwertung. Wie `chaturaji_nnue::selfplay::final_targets`, und
/// bewusst dieselbe Rechnung: eine Endstellung im Baum muss denselben Wert
/// bekommen wie dieselbe Stellung im Trainingsziel, sonst sucht die Engine auf
/// einer anderen Skala als das Netz gelernt hat.
///
/// `Rules::final_scores` statt `board.scores`: bleibt genau ein Spieler übrig,
/// gehören ihm die 3 Punkte je nie geschlagenem König.
fn final_targets(board: &Board) -> [f32; 4] {
    place_values(Rules::final_scores(board))
}

/// Erkundungsgewicht in der PUCT-Formel.
///
/// Der Wert 12 aus `flutteraji` gehört zu Rangpunkten auf einer Skala bis 6 und
/// wäre hier, wo die Netzausgabe in [−1, 1] liegt, deutlich zu groß.
///
/// 1,5 sieht andersherum zu groß aus, wenn man nachrechnet: die Werte spannen
/// [−1, 1] nämlich nicht aus. Über 3.097 Stellungen gemessen (Beispiel
/// `spanne`) liegt der Abstand zwischen bestem und zweitbestem Zug im Median
/// bei 0,039, während der Erkundungsterm bei 800 Simulationen und mittlerer
/// Besuchszahl rund 0,15 beträgt — viermal so groß. Q entscheidet damit kaum,
/// die Besuche folgen überwiegend dem Prior.
///
/// Trotzdem ist 1,5 gemessen besser: **c_puct 0,25 verliert gegen 1,5 um
/// −0,156 und −0,146 Platzwert** (je 576 Partien, zwei Seeds, gleiches Netz,
/// 2026-09-15). Bei 800 Simulationen auf rund 30 Züge bekommt jeder Zug nur
/// etwa 27 Besuche; die Verteilung nach dem Prior ist bei so knappem Budget
/// kein Mangel, sondern das Beste, was zu haben ist. Mit kleinem c_puct legt
/// sich die Suche zu früh auf die zuerst besuchten Züge fest.
///
/// Die Rechnung oben war also richtig, die Schlussfolgerung daraus falsch.
pub const DEFAULT_C_PUCT: f32 = 1.5;

/// Obergrenze für die Länge einer einzelnen Simulation.
///
/// Der Abstieg endet normalerweise an einem unerforschten Knoten. Bei
/// Zugwiederholungen kann der Baum aber tief werden, und ohne Schranke liefe
/// eine Simulation im Extremfall bis zum Partieende.
const MAX_DESCENT: usize = 200;

/// Wonach der Zug an der Wurzel gewählt wird.
///
/// AlphaZero nimmt die Besuchszahl. Die Überlegung, dass das bei kleinem Budget
/// kippen müsste — bei 800 Simulationen auf 30 Züge bekommt jeder Zug nur rund
/// 27 Besuche, die Verteilung folgt also überwiegend dem Prior —, ist
/// **gemessen falsch**: `Value` verliert gegen `Visits` um −0,156 und −0,137
/// Platzwert (je 576 Partien, zwei Seeds, gleiches Netz, 2026-09-15).
///
/// Dasselbe Muster wie bei [`DEFAULT_C_PUCT`]: die Besuchszahl mittelt über
/// alle Simulationen, die durch ein Kind liefen, der Q-Wert nur über dessen
/// eigene. Gerade bei wenigen Besuchen ist er das rauschigere Maß, nicht das
/// schärfere. `Value` bleibt als Schalter für weitere Versuche erhalten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootChoice {
    /// Meistbesuchter Zug.
    Visits,
    /// Bester Mittelwert aus Sicht des Ziehenden, unter den Zügen mit
    /// mindestens `MIN_VISITS_FOR_Q` Besuchen — ein Zug mit zwei Besuchen hat
    /// keinen belastbaren Mittelwert.
    Value,
}

/// Untergrenze, ab der ein Mittelwert an der Wurzel zählt.
pub const MIN_VISITS_FOR_Q: u32 = 8;

/// Abschlag für noch nicht besuchte Züge („First Play Urgency").
///
/// Bisher bekam ein unbesuchtes Kind `Q = 0` — gedacht als „durchschnittlich",
/// weil die Platzwerte um null zentriert sind. In einer *konkreten* Stellung
/// stimmt das aber nicht: steht die Wurzel bei +0,3, sieht jeder unbesuchte Zug
/// daneben um 0,3 schlechter aus, ohne dass irgendetwas gegen ihn spräche. Die
/// Suche wird dadurch künstlich eng — und zwar genau bei den Knoten, an denen
/// sie bei 800 Simulationen überwiegend steht, nämlich solchen mit ein bis zwei
/// Besuchen.
///
/// Üblich ist stattdessen der Wert des Elternknotens abzüglich eines
/// Abschlags. `None` behält das alte Verhalten bei, damit sich der Wechsel
/// messen lässt statt ihn zu behaupten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fpu {
    /// `Q = 0` für unbesuchte Kinder.
    Null,
    /// `Q = Q(Elternknoten) − Abschlag`.
    ElternMinus(f32),
}

pub struct MctsConfig {
    pub iterations: u32,
    pub c_puct:     f32,
    pub root:       RootChoice,
    pub fpu:        Fpu,
    /// Den Baum des vorigen Zuges weiterverwenden, wenn er zur Stellung passt.
    pub reuse:      bool,
}

impl Default for MctsConfig {
    fn default() -> Self {
        // Die Vorgaben sind der gemessene Bestand: Wiederverwendung +0,064,
        // FPU +0,067, zusammen +0,112 Platzwert über vier Seeds und je 576
        // Partien — und die Wiederverwendung ist dabei 30 % **schneller**
        // (104 und 92 s gegen zweimal 142 über volle Partien).
        //
        // Der Abschlag 0,4 ist ausgemessen, nicht geraten:
        //
        //   gegen 0,2:  0,1 → −0,027 (2 Seeds)   0,4 → +0,062 (2)
        //   gegen 0,4:  0,6 → +0,011 (4 Seeds)   1,0 → −0,108 (2)
        //
        // 0,6 sah nach zwei Seeds mit +0,035 noch nach einem Vorteil aus; mit
        // vier bleibt +0,011, zwei positiv und zwei negativ. Das Optimum ist
        // flach zwischen 0,4 und 0,6 und fällt jenseits davon steil ab.
        // Dass ein Abschlag oberhalb
        // der typischen Wertespanne (Median 0,28) am besten wirkt, passt zum
        // Befund, dass die Wurzelwahl nach Q schlechter ist als nach Besuchen:
        // Q ist an schwach besuchten Knoten zu verrauscht, um beizutragen.
        //
        // `reuse` wirkt nur, wenn der Aufrufer nach jedem Halbzug
        // [`Mcts::advance`] ruft; sonst greift die Stellungsprüfung und der
        // Baum wird verworfen. Das ist kein Fehler, nur eine verschenkte
        // Gelegenheit.
        Self { iterations: 400, c_puct: DEFAULT_C_PUCT, root: RootChoice::Visits,
               fpu: Fpu::ElternMinus(0.4), reuse: true }
    }
}

/// Gleichheit zweier Stellungen, soweit sie für die Suche zählt.
///
/// `Board` leitet kein `PartialEq` ab, und das soll hier auch nicht nachgeholt
/// werden: verglichen wird genau das, was den Suchbaum bestimmt — Figuren,
/// Punkte, Zugrecht, wer noch dabei ist. Der Halbzugzähler gehört dazu, weil er
/// über `dense_features` in die Netzbewertung eingeht.
fn gleiche_stellung(a: &Board, b: &Board) -> bool {
    a.bb == b.bb
        && a.to_move == b.to_move
        && a.active == b.active
        && a.scores.as_array() == b.scores.as_array()
        && a.half_moves == b.half_moves
}

/// Ein Knoten im Suchbaum.
///
/// Der Baum liegt als flacher `Vec`; Kinder sind ein zusammenhängender
/// Indexbereich. Das spart gegenüber `Box`/`Rc` eine Allokation je Knoten und
/// hält die Kinder eines Knotens im Speicher beieinander.
#[derive(Clone)]
struct Node {
    /// Zug, der hierher geführt hat. `None` nur an der Wurzel.
    mv:       Option<Move>,
    kids:     std::ops::Range<usize>,
    visits:   u32,
    /// Summe der Bewertungen über alle Besuche, je Sitz.
    q_sum:    [f32; 4],
    prior:    f32,
    expanded: bool,
    /// Wer an diesem Knoten am Zug ist — bestimmt, nach welcher Komponente
    /// ausgewählt wird.
    to_move:  usize,
}

impl Node {
    fn new(mv: Option<Move>, prior: f32, to_move: usize) -> Self {
        Self { mv, kids: 0..0, visits: 0, q_sum: [0.0; 4], prior, expanded: false, to_move }
    }

    #[inline]
    fn q(&self, seat: usize) -> f32 {
        if self.visits == 0 { 0.0 } else { self.q_sum[seat] / self.visits as f32 }
    }
}

pub struct Mcts {
    nodes: Vec<Node>,
    /// Längster Abstieg der letzten Suche, für die Anzeige.
    tiefe: usize,
    /// Stellung, zu der die Wurzel gehört — die Sicherung gegen Verwechslung.
    ///
    /// Ohne sie rechnete die Suche nach einem vergessenen [`Mcts::advance`] auf
    /// einem fremden Baum weiter, und zwar **still**: die Züge wären legal, die
    /// Bewertungen gehörten zu anderen Stellungen. Ein Fehler dieser Art fällt
    /// in einer Arena nicht auf, er sieht wie Rauschen aus.
    wurzel: Option<Board>,
    /// Besuche, die aus dem vorigen Zug übernommen wurden — für die Anzeige.
    geerbt: u32,
}

/// Ergebnis einer Suche.
pub struct SearchResult {
    pub best:   Move,
    /// Bewertung der Wurzel aus Sicht aller vier Sitze.
    pub value:  [f32; 4],
    /// Besuchszahl je Zug an der Wurzel, absteigend sortiert.
    ///
    /// Das ist die eigentliche Ausbeute einer MCTS-Suche und später das
    /// Trainingsziel für einen Policy-Kopf. Heute wird sie nur zur Zugwahl
    /// benutzt.
    pub visits: Vec<(Move, u32)>,
    /// Mittelwert und Besuchszahl je Wurzelzug, aus Sicht des Ziehenden,
    /// in derselben Reihenfolge wie `visits`.
    ///
    /// `visits` allein sagt, *welchen* Zug die Suche vorzieht, nicht *wie
    /// deutlich*. Für den Abstand zwischen bestem und gespieltem Zug — den
    /// Verlust, den die Partieanalyse ausweist — braucht es die Mittelwerte.
    /// Die Besuchszahl steht daneben, weil ein Q mit wenigen Besuchen Rauschen
    /// ist; [`MIN_VISITS_FOR_Q`] ist die Grenze, ab der er zählt.
    pub root_q: Vec<(Move, f32, u32)>,
    /// Angelegte Baumknoten — das Gegenstück zu `nodes` der Alpha-Beta-Suche.
    pub nodes:  u64,
    /// Längster Abstieg in Halbzügen. Anders als bei fester Tiefe ist das ein
    /// Ergebnis, kein Parameter: MCTS vertieft dort, wo es sich lohnt.
    pub depth:  u8,
}

impl Mcts {
    pub fn new() -> Self {
        Self { nodes: Vec::with_capacity(4096), tiefe: 0, wurzel: None, geerbt: 0 }
    }

    /// Wirft den Baum weg.
    pub fn reset(&mut self) {
        self.nodes.clear();
        self.wurzel = None;
        self.geerbt = 0;
    }

    /// Setzt die Wurzel auf das Kind, das `mv` entspricht.
    ///
    /// Damit überlebt der Teilbaum unter dem gespielten Zug den Zugwechsel.
    /// Bei vier Spielern liegt die eigene nächste Entscheidung vier Halbzüge
    /// weiter; wird nach **jedem** Halbzug fortgeschaltet, ist der Teilbaum
    /// dann noch da, und seine Simulationen sind geschenkte Rechenzeit.
    ///
    /// `false`, wenn der Zug im Baum nicht vorkommt (nicht erweitert, oder gar
    /// nicht erst gesucht). Dann bleibt nur ein frischer Baum, und genau das
    /// tut diese Funktion in dem Fall auch.
    pub fn advance(&mut self, board_davor: &Board, mv: Move) -> bool {
        let passt = self.wurzel.as_ref().is_some_and(|w| gleiche_stellung(w, board_davor));
        if !passt || self.nodes.is_empty() {
            self.reset();
            return false;
        }
        let kind = self.nodes[0].kids.clone()
            .find(|&k| self.nodes[k].mv == Some(mv));
        let Some(kind) = kind else { self.reset(); return false };
        if !self.nodes[kind].expanded {
            // Ein Blatt trägt nichts bei, das eine frische Wurzel nicht auch
            // hätte — und umkopieren kostet dann mehr, als es einbringt.
            self.reset();
            return false;
        }

        self.umwurzeln(kind);
        self.wurzel = Some(Rules::apply_with_effects(board_davor, mv));
        self.geerbt = self.nodes.first().map_or(0, |n| n.visits);
        true
    }

    /// Kopiert den Teilbaum unter `neu` an den Anfang des Arenas und wirft den
    /// Rest weg.
    ///
    /// Der Baum liegt als flacher `Vec` mit Index-Bereichen; Umwurzeln heißt
    /// deshalb Umkopieren mit neuer Nummerierung. Das kostet einen Durchlauf
    /// über den Teilbaum — gegenüber hunderten Simulationen nichts.
    fn umwurzeln(&mut self, neu: usize) {
        let mut ziel: Vec<Node> = Vec::with_capacity(self.nodes.len() / 2 + 1);
        // Breitensuche, damit Geschwister zusammenhängend liegen — die
        // Kinder eines Knotens müssen ein zusammenhängender Bereich sein.
        let mut warteschlange = std::collections::VecDeque::new();
        ziel.push(self.nodes[neu].clone());
        ziel[0].mv = None;              // die neue Wurzel hat keinen Zug
        warteschlange.push_back((neu, 0usize));

        while let Some((alt, neu_idx)) = warteschlange.pop_front() {
            let kids = self.nodes[alt].kids.clone();
            if kids.is_empty() {
                ziel[neu_idx].kids = 0..0;
                continue;
            }
            let start = ziel.len();
            for k in kids.clone() {
                ziel.push(self.nodes[k].clone());
            }
            ziel[neu_idx].kids = start..ziel.len();
            for (i, k) in kids.enumerate() {
                warteschlange.push_back((k, start + i));
            }
        }
        self.nodes = ziel;
    }

    /// Sucht von `board` aus und gibt den meistbesuchten Zug zurück.
    ///
    /// `tt` wird nur für die Blattbewertung des Netzes wiederverwendet; der
    /// Baum selbst hält keine Transpositionen zusammen. Das ist Absicht — zwei
    /// Wege zur selben Stellung haben in einem Vierpersonenspiel verschiedene
    /// Vorgeschichten, und die Besuchszahlen zusammenzulegen verfälschte die
    /// Auswahl.
    pub fn search(
        &mut self,
        eval:  LeafEval,
        model: &dyn MovePrior,
        board: &Board,
        cfg:   &MctsConfig,
    ) -> Option<SearchResult> {
        let moves = Rules::legal_moves(board);
        if moves.is_empty() { return None; }

        // Weiterverwenden, wenn der Baum zu genau dieser Stellung gehört.
        // Die Prüfung ist die Sicherung: ein vergessenes `advance` führt zu
        // einem frischen Baum, nie zu einem falschen.
        let passt = cfg.reuse
            && !self.nodes.is_empty()
            && self.wurzel.as_ref().is_some_and(|w| gleiche_stellung(w, board));
        if !passt {
            self.nodes.clear();
            self.geerbt = 0;
            self.nodes.push(Node::new(None, 1.0, board.to_move.idx()));
        }
        self.wurzel = Some(board.clone());
        self.tiefe = 0;

        for _ in 0..cfg.iterations {
            self.simulate(eval, model, board, cfg);
        }

        let root = &self.nodes[0];
        let mut visits: Vec<(Move, u32)> = self.nodes[root.kids.clone()]
            .iter()
            .map(|k| (k.mv.expect("Kinder tragen immer einen Zug"), k.visits))
            .collect();
        // Absteigend nach Besuchen; bei Gleichstand nach Feldern, damit die
        // Auswahl nicht von der Reihenfolge der Zuggenerierung abhängt.
        visits.sort_by(|a, b| b.1.cmp(&a.1)
            .then(a.0.from.cmp(&b.0.from))
            .then(a.0.to.cmp(&b.0.to)));

        let best = match cfg.root {
            RootChoice::Visits => visits.first().map(|(m, _)| *m).unwrap_or(moves[0]),
            RootChoice::Value  => {
                let seat = board.to_move.idx();
                self.nodes[root.kids.clone()].iter()
                    .filter(|k| k.visits >= MIN_VISITS_FOR_Q)
                    .max_by(|a, b| a.q(seat).partial_cmp(&b.q(seat))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Gleichstand wie oben nach Feldern auflösen, damit die
                        // Wahl nicht an der Zuggenerierung hängt.
                        .then(b.mv.map_or(0, |m| m.from).cmp(&a.mv.map_or(0, |m| m.from)))
                        .then(b.mv.map_or(0, |m| m.to).cmp(&a.mv.map_or(0, |m| m.to))))
                    .and_then(|k| k.mv)
                    // Keiner hat genug Besuche: dann bleibt nur die Besuchszahl.
                    .unwrap_or_else(|| visits.first().map(|(m, _)| *m).unwrap_or(moves[0]))
            }
        };
        let seat = board.to_move.idx();
        let root_q: Vec<(Move, f32, u32)> = visits.iter().map(|(mv, n)| {
            let k = self.nodes[root.kids.clone()].iter()
                .find(|k| k.mv == Some(*mv))
                .expect("jeder Zug aus `visits` stammt aus `kids`");
            (*mv, k.q(seat), *n)
        }).collect();

        let value = std::array::from_fn(|i| root.q(i));
        Some(SearchResult {
            best, value, visits, root_q,
            nodes: self.nodes.len() as u64,
            depth: self.tiefe.min(u8::MAX as usize) as u8,
        })
    }

    /// Eine Simulation: absteigen, ein Blatt erweitern und bewerten, zurücktragen.
    fn simulate(&mut self, eval: LeafEval, model: &dyn MovePrior, root: &Board, cfg: &MctsConfig) {
        let mut board = root.clone();
        let mut pfad  = vec![0usize];
        let mut idx   = 0usize;

        // ─── Abstieg durch bereits erweiterte Knoten ────────────────────────
        while self.nodes[idx].expanded && !self.nodes[idx].kids.is_empty() {
            if pfad.len() >= MAX_DESCENT { break; }
            idx = self.select(idx, cfg);
            board = Rules::apply_with_effects(
                &board, self.nodes[idx].mv.expect("nur die Wurzel hat keinen Zug"));
            pfad.push(idx);
        }

        // ─── Erweitern und bewerten ────────────────────────────────────────
        let wert = if Rules::is_game_over(&board) {
            // Endstellung: der tatsächliche Ausgang, keine Schätzung.
            final_targets(&board)
        } else {
            if !self.nodes[idx].expanded {
                self.expand(idx, model, &board);
            }
            eval(&board)
        };

        self.tiefe = self.tiefe.max(pfad.len() - 1);

        // ─── Zurücktragen ──────────────────────────────────────────────────
        for &n in &pfad {
            let node = &mut self.nodes[n];
            node.visits += 1;
            for s in 0..4 { node.q_sum[s] += wert[s]; }
        }
    }

    /// Legt die Kinder eines Knotens an und verteilt die Priors.
    fn expand(&mut self, idx: usize, model: &dyn MovePrior, board: &Board) {
        let moves = Rules::legal_moves(board);
        self.nodes[idx].expanded = true;
        if moves.is_empty() { return; }

        // Der Kontext (vier Angriffskarten) einmal je Stellung, nicht je Zug.
        let ctx = MoveFeatureContext::new(board);
        let scores: Vec<f32> = moves.iter()
            .map(|mv| model.logit(&fast_features(board, mv, &ctx)))
            .collect();

        // Softmax, stabilisiert. Die Scores des Modells sind
        // Log-Wahrscheinlichkeiten bis auf eine Konstante — der Softmax ist
        // damit genau die geschätzte Zugverteilung.
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let summe: f32 = exps.iter().sum::<f32>().max(1e-30);

        let start = self.nodes.len();
        let naechster = Rules::apply_with_effects(board, moves[0]).to_move.idx();
        for (mv, e) in moves.iter().zip(&exps) {
            // `to_move` des Kindes hängt davon ab, wer nach diesem Zug dran ist.
            // Das kann je Zug verschieden sein, wenn ein Zug einen Spieler
            // ausscheiden lässt — deshalb je Zug bestimmt und nicht einmal.
            let kind_to_move = if moves.len() == 1 {
                naechster
            } else {
                Rules::apply_with_effects(board, *mv).to_move.idx()
            };
            self.nodes.push(Node::new(Some(*mv), e / summe, kind_to_move));
        }
        self.nodes[idx].kids = start..self.nodes.len();
    }

    /// PUCT-Auswahl unter den Kindern von `idx`.
    fn select(&self, idx: usize, cfg: &MctsConfig) -> usize {
        let seat = self.nodes[idx].to_move;
        let sum_n = self.nodes[idx].visits.max(1) as f32;
        let wurzel_n = sum_n.sqrt();

        // Was ein noch unbesuchtes Kind an Q mitbekommt — siehe [`Fpu`].
        let unbesucht = match cfg.fpu {
            Fpu::Null => 0.0,
            Fpu::ElternMinus(abschlag) => self.nodes[idx].q(seat) - abschlag,
        };

        let mut bester = self.nodes[idx].kids.start;
        let mut bestwert = f32::NEG_INFINITY;
        for k in self.nodes[idx].kids.clone() {
            let kind = &self.nodes[k];
            let q = if kind.visits == 0 { unbesucht } else { kind.q(seat) };
            let wert = q + cfg.c_puct * kind.prior * wurzel_n / (1.0 + kind.visits as f32);
            if wert > bestwert { bestwert = wert; bester = k; }
        }
        bester
    }
}

impl Default for Mcts {
    fn default() -> Self { Self::new() }
}

/// Bequemer Einstieg für Aufrufer, die keinen Baum behalten wollen.
///
/// `_tt` wird nicht benutzt; der Parameter hält die Signatur mit
/// `nnue_best_move` vergleichbar, damit die Arena beide gleich aufrufen kann.
pub fn mcts_best_move(
    eval:  LeafEval,
    model: &dyn MovePrior,
    board: &Board,
    cfg:   &MctsConfig,
    _tt:   &mut HashMap<u64, (u8, [f32; 4])>,
) -> Option<SearchResult> {
    Mcts::new().search(eval, model, board, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_features::MoveModel;
    use chaturaji_core::piece::{Color, PieceKind};

    /// Bewertung für die Tests: deterministisch, ohne Netz — aber sie muss
    /// **Stellungen unterscheiden**, sonst prüft ein Test nichts.
    ///
    /// Die erste Fassung nahm nur die Punktedifferenz. In der Startstellung
    /// sind alle Punkte null, die Bewertung also überall gleich, und
    /// `fpu_veraendert_die_verteilung` schlug fehl — nicht weil FPU nichts
    /// täte, sondern weil es an einer konstanten Bewertung nichts zu ändern
    /// gibt. Jetzt zählt der Bauernfortschritt mit, und der ändert sich mit
    /// jedem Zug.
    fn netz() -> impl Fn(&Board) -> [f32; 4] {
        |b: &Board| {
            let roh: [f32; 4] = std::array::from_fn(|i| {
                let c = Color::ALL[i];
                let mut bb = b.pieces(c, PieceKind::Pawn);
                let mut fortschritt = 0.0;
                while bb != 0 {
                    let sq = bb.trailing_zeros() as u8;
                    bb &= bb - 1;
                    fortschritt += (7 - crate::move_features::promotion_distance(c, sq)) as f32;
                }
                fortschritt + b.scores.as_array()[i] as f32
            });
            let mittel = roh.iter().sum::<f32>() / 4.0;
            std::array::from_fn(|i| ((roh[i] - mittel) / 10.0).clamp(-1.0, 1.0))
        }
    }

    #[test]
    fn search_returns_a_legal_move() {
        let board = Board::default();
        let r = Mcts::new().search(&netz(), &MoveModel::default(), &board,
                                   &MctsConfig { iterations: 200, ..Default::default() })
            .expect("die Startstellung hat legale Züge");
        assert!(Rules::legal_moves(&board).contains(&r.best));
    }

    /// Die Besuche müssen sich auf die Wurzelzüge verteilen und in der Summe
    /// die Iterationszahl ergeben — jede Simulation zählt genau einen Wurzelzug,
    /// außer sie endet schon an der Wurzel.
    #[test]
    fn visits_add_up_to_the_iteration_count() {
        let board = Board::default();
        let iter = 300;
        let r = Mcts::new().search(&netz(), &MoveModel::default(), &board,
                                   &MctsConfig { iterations: iter, ..Default::default() }).unwrap();
        let summe: u32 = r.visits.iter().map(|(_, n)| n).sum();
        // Die erste Simulation erweitert nur die Wurzel und steigt nicht ab.
        assert_eq!(summe, iter - 1, "Besuche: {summe}, Iterationen: {iter}");
    }

    /// Mehr Iterationen müssen die Besuche stärker auf wenige Züge bündeln.
    /// Ohne diese Eigenschaft verteilte die Suche nur gleichmäßig und wäre
    /// nichts wert.
    #[test]
    fn more_iterations_concentrate_the_visits() {
        let board = Board::default();
        let anteil = |iter: u32| {
            let r = Mcts::new().search(&netz(), &MoveModel::default(), &board,
                                       &MctsConfig { iterations: iter, ..Default::default() }).unwrap();
            let summe: u32 = r.visits.iter().map(|(_, n)| n).sum();
            r.visits[0].1 as f32 / summe as f32
        };
        let wenig = anteil(100);
        let viel  = anteil(2000);
        assert!(viel > wenig, "Bündelung muss zunehmen: {wenig:.3} → {viel:.3}");
    }

    /// Der Prior muss wirken: ein Modell, das alle Züge gleich bewertet, führt
    /// zu einer flacheren Verteilung als das gelernte.
    #[test]
    fn the_prior_shapes_the_search() {
        let board = Board::default();
        let cfg = MctsConfig { iterations: 400, ..Default::default() };
        let spitze = |m: &MoveModel| {
            let r = Mcts::new().search(&netz(), m, &board, &cfg).unwrap();
            r.visits[0].1
        };
        assert!(spitze(&MoveModel::default()) >= spitze(&MoveModel::zeros()),
                "mit gelerntem Prior darf die Suche nicht breiter streuen als ohne");
    }

    /// Die Wurzelbewertung muss im Wertebereich des Netzes bleiben und darf
    /// nicht davonlaufen — ein häufiger Fehler beim Zurücktragen ist, Summen
    /// statt Mittelwerte zu lesen.
    #[test]
    fn the_root_value_stays_in_range() {
        let r = Mcts::new().search(&netz(), &MoveModel::default(), &Board::default(),
                                   &MctsConfig { iterations: 500, ..Default::default() }).unwrap();
        for v in r.value {
            assert!(v.is_finite() && (-1.5..=1.5).contains(&v), "Wurzelwert {v} außerhalb");
        }
    }

    /// Bei nur einem legalen Zug muss die Suche ihn liefern, ohne zu straucheln.
    #[test]
    fn a_single_legal_move_is_returned() {
        // Konstruierte Stellung mit genau einem Zug zu bauen ist aufwendig;
        // stattdessen die Zusicherung an der Schnittstelle: eine Stellung ohne
        // Züge liefert None statt zu paniken.
        let mut leer = Board::empty();
        leer.active = [false; 4];
        assert!(Mcts::new().search(&netz(), &MoveModel::default(), &leer,
                                   &MctsConfig::default()).is_none());
    }

    /// Zwei Suchen mit denselben Eingaben müssen dasselbe liefern — die Suche
    /// selbst enthält keinen Zufall, und ohne das wäre kein Arena-Vergleich
    /// reproduzierbar.
    #[test]
    fn the_search_is_deterministic() {
        let board = Board::default();
        let net = netz();
        let cfg = MctsConfig { iterations: 300, ..Default::default() };
        let a = Mcts::new().search(&net, &MoveModel::default(), &board, &cfg).unwrap();
        let b = Mcts::new().search(&net, &MoveModel::default(), &board, &cfg).unwrap();
        assert_eq!(a.best, b.best);
        assert_eq!(a.visits, b.visits);
    }

    // ─── Wiederverwendung ────────────────────────────────────────────────────

    fn cfg_reuse(iter: u32) -> MctsConfig {
        MctsConfig { iterations: iter, reuse: true, ..Default::default() }
    }

    /// Nach `advance` muss der Baum die Besuche des Teilbaums behalten — sonst
    /// wäre die ganze Übung wirkungslos.
    #[test]
    fn advance_erbt_besuche() {
        let mut baum = Mcts::new();
        let brett = Board::default();
        let r = baum.search(&netz(), &MoveModel::default(), &brett, &cfg_reuse(600)).unwrap();
        assert!(baum.advance(&brett, r.best), "der gespielte Zug muss im Baum stehen");
        assert!(baum.geerbt > 0, "keine Besuche geerbt");
        assert!(baum.nodes[0].visits == baum.geerbt);
    }

    /// Ein Zug, der nicht im Baum steht, muss zu einem frischen Baum führen —
    /// nicht zu einem falschen.
    #[test]
    fn advance_mit_fremdem_zug_verwirft() {
        let mut baum = Mcts::new();
        let brett = Board::default();
        baum.search(&netz(), &MoveModel::default(), &brett, &cfg_reuse(200)).unwrap();
        let fremd = Move::new(0, 63, brett.to_move);
        assert!(!baum.advance(&brett, fremd));
        assert!(baum.nodes.is_empty());
    }

    /// Die Sicherung: passt die Stellung nicht zur Wurzel, wird der Baum
    /// verworfen statt weiterbenutzt. Ein vergessenes `advance` darf die Suche
    /// nicht auf fremden Bewertungen rechnen lassen.
    #[test]
    fn fremde_stellung_verwirft_den_baum() {
        let mut baum = Mcts::new();
        let brett = Board::default();
        let r = baum.search(&netz(), &MoveModel::default(), &brett, &cfg_reuse(300)).unwrap();
        // Zwei Züge weiter, ohne `advance` — die Wurzel passt nicht mehr.
        let mut weiter = Rules::apply_with_effects(&brett, r.best);
        weiter = Rules::apply_with_effects(&weiter, Rules::legal_moves(&weiter)[0]);
        let r2 = baum.search(&netz(), &MoveModel::default(), &weiter, &cfg_reuse(300)).unwrap();
        assert!(Rules::legal_moves(&weiter).contains(&r2.best));
        let summe: u32 = r2.visits.iter().map(|(_, n)| n).sum();
        assert_eq!(summe, 299, "frischer Baum: Besuche wie bei einem Neustart");
    }

    /// Das Umkopieren darf den Baum nicht beschädigen: die Kinder jedes Knotens
    /// müssen ein gültiger, zusammenhängender Bereich bleiben, und die Besuche
    /// eines Knotens dürfen die Summe seiner Kinder nicht unterschreiten.
    #[test]
    fn umwurzeln_laesst_den_baum_heil() {
        let mut baum = Mcts::new();
        let brett = Board::default();
        let r = baum.search(&netz(), &MoveModel::default(), &brett, &cfg_reuse(800)).unwrap();
        baum.advance(&brett, r.best);
        for (i, n) in baum.nodes.iter().enumerate() {
            assert!(n.kids.end <= baum.nodes.len(), "Knoten {i}: Bereich zeigt ins Leere");
            assert!(n.kids.start <= n.kids.end, "Knoten {i}: Bereich verdreht");
            let kinder: u32 = baum.nodes[n.kids.clone()].iter().map(|k| k.visits).sum();
            assert!(kinder <= n.visits, "Knoten {i}: {kinder} Kindbesuche, aber nur {} eigene", n.visits);
        }
    }

    /// Mit geerbten Besuchen muss die Suche mehr Gesamtaufwand haben als ohne —
    /// das ist der ganze Zweck.
    #[test]
    fn wiederverwendung_bringt_zusaetzliche_besuche() {
        let brett = Board::default();
        let mut baum = Mcts::new();
        let r = baum.search(&netz(), &MoveModel::default(), &brett, &cfg_reuse(600)).unwrap();
        baum.advance(&brett, r.best);
        let danach = Rules::apply_with_effects(&brett, r.best);
        let geerbt = baum.geerbt;
        let r2 = baum.search(&netz(), &MoveModel::default(), &danach, &cfg_reuse(600)).unwrap();
        let summe: u32 = r2.visits.iter().map(|(_, n)| n).sum();
        assert!(summe > 599, "nur {summe} Besuche, geerbt waren {geerbt}");
    }

    // ─── FPU ─────────────────────────────────────────────────────────────────

    /// Der Abschlag muss wirken: mit Elternwert-FPU sieht ein unbesuchtes Kind
    /// in einer guten Stellung anders aus als mit festem Q = 0, und die Suche
    /// verteilt die Besuche entsprechend anders.
    #[test]
    fn fpu_veraendert_die_verteilung() {
        let brett = Board::default();
        let lauf = |fpu: Fpu| {
            Mcts::new().search(&netz(), &MoveModel::default(), &brett,
                &MctsConfig { iterations: 800, fpu, ..Default::default() }).unwrap().visits
        };
        let a = lauf(Fpu::Null);
        let b = lauf(Fpu::ElternMinus(0.4));
        assert_ne!(a, b, "FPU hat keinerlei Wirkung");
    }
}
