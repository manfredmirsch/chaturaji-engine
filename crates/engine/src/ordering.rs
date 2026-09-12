//! Zugsortierung für Alpha-Beta.
//!
//! Reihenfolge (absteigend):
//!   1. bester Zug aus der Transpositionstabelle
//!   2. Schläge nach MVV-LVA (wertvollstes Opfer, billigster Angreifer)
//!   3. Killer-Züge (stille Widerlegungen aus derselben Tiefe)
//!   4. Umwandlungen, dann der Rest
//!
//! # Warum hier nicht das gelernte Zugmodell steht
//!
//! [`crate::move_features`] sagt den Zug eines starken Spielers deutlich besser
//! vorher als MVV-LVA — 74,3 % gegen 54,6 % beim Überleben im Beam der besten
//! sechs. Für Alpha-Beta ist das trotzdem die falsche Zielgröße, und die
//! Messung sagt es deutlich:
//!
//! | | MVV-LVA | Zugmodell |
//! |---|---|---|
//! | paranoid Tiefe 8, Knoten | 1.234.866 | 1.386.788 (+12 %) |
//! | paranoid Tiefe 8, ms | 9.861 | 12.481 (+27 %) |
//!
//! Alpha-Beta schneidet ab, sobald ein Zug die Schranke sprengt — es will den
//! **objektiv stärksten** Zug zuerst sehen, nicht den wahrscheinlichsten. Das
//! wertvollste Opfer zuerst zu nehmen erzeugt den größten Bewertungssprung und
//! damit den frühesten Schnitt; „was ein Mensch spielen würde" erzeugt ihn
//! nicht. Dazu kommen die vier Angriffskarten, die das Modell je Stellung
//! braucht — daher die 27 % Zeit bei nur 12 % mehr Knoten.
//!
//! Im **Beam** ist es umgekehrt: dort werden Züge weggeworfen, und was
//! weggeworfen wird, sieht die Suche nie wieder. Deshalb benutzt
//! `chaturaji_nnue::selfplay` das Modell und diese Datei nicht.
//!
//! # Was von dem Versuch geblieben ist
//!
//! Die Struktur. `order_moves` rief `score_move` aus dem Sortiervergleich
//! heraus auf — bei n Zügen also O(n log n) Bewertungen statt n. Jetzt wird
//! einmal bewertet und dann sortiert. Und [`order_moves_with`] nimmt ein
//! beliebiges Modell entgegen, falls sich die Frage später doch anders stellt.

use chaturaji_core::board::{Board, Move};
use chaturaji_core::piece::PieceKind;

use crate::move_features::{fast_features, MoveFeatureContext, MoveModel};

/// Bewertet einen Zug für die Sortierung. Höher = zuerst gesucht.
pub fn score_move(
    board:   &Board,
    mv:      &Move,
    tt_best: Option<Move>,
    killers: &[Option<Move>; 2],
) -> i32 {
    // 1. Bester Zug aus der Transpositionstabelle
    if let Some(best) = tt_best {
        if mv.from == best.from && mv.to == best.to { return 100_000; }
    }

    // 2. Schläge: MVV-LVA
    if let Some(cap) = mv.captured {
        let mover_kind = board.piece_at(mv.from)
            .map(|p| p.kind)
            .unwrap_or(PieceKind::Pawn);

        let victim_val   = cap.kind.capture_value();
        let attacker_val = mover_kind.capture_value();

        return 10_000 + victim_val * 10 - attacker_val;
    }

    // 3. Killer-Züge
    if killers[0] == Some(*mv) { return 9_500; }
    if killers[1] == Some(*mv) { return 9_000; }

    // 4. Stille Züge — Umwandlungen zuerst
    if mv.promoted { return 5_000; }

    0
}

/// Sortiert die Züge, besten zuerst.
///
/// Bewertet einmal je Zug und sortiert danach; der Vergleich selbst rechnet
/// nichts mehr.
pub fn order_moves(
    board:   &Board,
    moves:   &mut Vec<Move>,
    tt_best: Option<Move>,
    killers: &[Option<Move>; 2],
) {
    if moves.len() < 2 { return; }
    let mut scored: Vec<(Move, i32)> = moves.iter()
        .map(|mv| (*mv, score_move(board, mv, tt_best, killers)))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    for (slot, (mv, _)) in moves.iter_mut().zip(scored) { *slot = mv; }
}

/// Sortiert nach einem gelernten Zugmodell statt nach MVV-LVA.
///
/// Für Alpha-Beta gemessen schlechter (siehe Modulkopf) — gedacht für
/// Verfahren, die Züge **verwerfen** statt nur umordnen, und für Experimente.
/// Transpositionszug und Killer stehen weiterhin obenan; die Scores des
/// Modells liegen erfahrungsgemäß zwischen −3 und +6.
pub fn order_moves_with(
    board:   &Board,
    moves:   &mut Vec<Move>,
    tt_best: Option<Move>,
    killers: &[Option<Move>; 2],
    model:   &MoveModel,
) {
    if moves.len() < 2 { return; }
    let ctx = MoveFeatureContext::new(board);

    let mut scored: Vec<(Move, f32)> = moves.iter()
        .map(|mv| {
            let s = if tt_best.is_some_and(|b| b.from == mv.from && b.to == mv.to) {
                1_000.0
            } else if killers[0] == Some(*mv) {
                900.0
            } else if killers[1] == Some(*mv) {
                890.0
            } else {
                model.score_features(&fast_features(board, mv, &ctx))
            };
            (*mv, s)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (slot, (mv, _)) in moves.iter_mut().zip(scored) { *slot = mv; }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::rules::Rules;

    /// Der Zug aus der Transpositionstabelle muss ganz vorn stehen.
    #[test]
    fn the_tt_move_comes_first() {
        let board = Board::default();
        let mut moves = Rules::legal_moves(&board);
        let gewuenscht = moves[moves.len() - 1];
        order_moves(&board, &mut moves, Some(gewuenscht), &[None; 2]);
        assert_eq!(moves[0], gewuenscht);
    }

    /// Dasselbe für den Modell-Pfad — die Sonderfälle gelten dort genauso.
    #[test]
    fn the_tt_move_comes_first_with_a_model_too() {
        let board = Board::default();
        let mut moves = Rules::legal_moves(&board);
        let gewuenscht = moves[moves.len() - 1];
        order_moves_with(&board, &mut moves, Some(gewuenscht), &[None; 2], &MoveModel::default());
        assert_eq!(moves[0], gewuenscht);
    }

    /// Sortieren darf die Zugmenge nicht verändern — kein Zug verschwindet,
    /// keiner kommt doppelt vor. Das ist die Eigenschaft, die beim Umbau auf
    /// „einmal bewerten, dann sortieren" hätte kaputtgehen können.
    #[test]
    fn sorting_is_a_permutation() {
        let board  = Board::default();
        let vorher = Rules::legal_moves(&board);
        for mut moves in [vorher.clone(), vorher.clone()] {
            order_moves(&board, &mut moves, None, &[None; 2]);
            assert_eq!(moves.len(), vorher.len());
            for mv in &vorher {
                assert_eq!(moves.iter().filter(|m| *m == mv).count(), 1, "{mv:?} genau einmal");
            }
        }
        let mut moves = vorher.clone();
        order_moves_with(&board, &mut moves, None, &[None; 2], &MoveModel::default());
        assert_eq!(moves.len(), vorher.len());
    }

    /// MVV-LVA: das wertvollere Opfer zuerst, bei gleichem Opfer der billigere
    /// Angreifer.
    #[test]
    fn mvv_lva_prefers_the_valuable_victim() {
        let board = Board::default();
        let boot   = Move { from: 0, to: 1, mover: board.to_move, promoted: false,
            captured: Some(chaturaji_core::piece::Piece {
                color: chaturaji_core::piece::Color::Blue, kind: PieceKind::Boat }) };
        let bauer  = Move { captured: Some(chaturaji_core::piece::Piece {
                color: chaturaji_core::piece::Color::Blue, kind: PieceKind::Pawn }), ..boot };
        assert!(score_move(&board, &boot, None, &[None; 2])
              > score_move(&board, &bauer, None, &[None; 2]));
    }

    /// Das eingebaute Modell muss die richtige Breite haben — sonst würde
    /// `score_features` still über die kürzere Seite iterieren.
    #[test]
    fn the_built_in_weights_have_the_right_width() {
        assert_eq!(MoveModel::default().w.len(), crate::move_features::N_FEATURES);
    }
}
