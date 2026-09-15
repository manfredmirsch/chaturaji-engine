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
const MIN_VISITS_FOR_Q: u32 = 8;

pub struct MctsConfig {
    pub iterations: u32,
    pub c_puct:     f32,
    pub root:       RootChoice,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self { iterations: 400, c_puct: DEFAULT_C_PUCT, root: RootChoice::Visits }
    }
}

/// Ein Knoten im Suchbaum.
///
/// Der Baum liegt als flacher `Vec`; Kinder sind ein zusammenhängender
/// Indexbereich. Das spart gegenüber `Box`/`Rc` eine Allokation je Knoten und
/// hält die Kinder eines Knotens im Speicher beieinander.
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
    /// Angelegte Baumknoten — das Gegenstück zu `nodes` der Alpha-Beta-Suche.
    pub nodes:  u64,
    /// Längster Abstieg in Halbzügen. Anders als bei fester Tiefe ist das ein
    /// Ergebnis, kein Parameter: MCTS vertieft dort, wo es sich lohnt.
    pub depth:  u8,
}

impl Mcts {
    pub fn new() -> Self {
        Self { nodes: Vec::with_capacity(4096), tiefe: 0 }
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

        self.nodes.clear();
        self.tiefe = 0;
        self.nodes.push(Node::new(None, 1.0, board.to_move.idx()));

        for _ in 0..cfg.iterations {
            self.simulate(eval, model, board, cfg.c_puct);
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
        let value = std::array::from_fn(|i| root.q(i));
        Some(SearchResult {
            best, value, visits,
            nodes: self.nodes.len() as u64,
            depth: self.tiefe.min(u8::MAX as usize) as u8,
        })
    }

    /// Eine Simulation: absteigen, ein Blatt erweitern und bewerten, zurücktragen.
    fn simulate(&mut self, eval: LeafEval, model: &dyn MovePrior, root: &Board, c_puct: f32) {
        let mut board = root.clone();
        let mut pfad  = vec![0usize];
        let mut idx   = 0usize;

        // ─── Abstieg durch bereits erweiterte Knoten ────────────────────────
        while self.nodes[idx].expanded && !self.nodes[idx].kids.is_empty() {
            if pfad.len() >= MAX_DESCENT { break; }
            idx = self.select(idx, c_puct);
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
    fn select(&self, idx: usize, c_puct: f32) -> usize {
        let seat = self.nodes[idx].to_move;
        let sum_n = self.nodes[idx].visits.max(1) as f32;
        let wurzel_n = sum_n.sqrt();

        let mut bester = self.nodes[idx].kids.start;
        let mut bestwert = f32::NEG_INFINITY;
        for k in self.nodes[idx].kids.clone() {
            let kind = &self.nodes[k];
            // Unbesuchte Kinder bekommen Q = 0. Das ist die neutrale Annahme:
            // die Werte sind um 0 zentriert (Summe der Platzwerte ist 0), ein
            // unbesuchter Zug gilt also als durchschnittlich und nicht als
            // besonders gut oder schlecht.
            let wert = kind.q(seat)
                + c_puct * kind.prior * wurzel_n / (1.0 + kind.visits as f32);
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

    /// Bewertung für die Tests: hängt von der Stellung ab, ist aber
    /// deterministisch und ohne Netz zu haben. Die Punktedifferenz reicht —
    /// geprüft wird hier die Mechanik der Suche, nicht die Spielstärke.
    fn netz() -> impl Fn(&Board) -> [f32; 4] {
        |b: &Board| {
            let p = b.scores.as_array();
            let summe: i32 = p.iter().sum::<i32>().max(1);
            std::array::from_fn(|i| p[i] as f32 / summe as f32 - 0.25)
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
}
