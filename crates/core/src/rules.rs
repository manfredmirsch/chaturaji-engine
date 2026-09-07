//! High-level rule enforcement for chess.com Chaturaji.
//!
//! This layer sits above movegen and handles:
//!   • Double-check (+1) and triple-check (+5) scoring bonus
//!   • Aufgabe und Zeitüberschreitung
//!   • Game-over detection
//!   • Filtering out moves that leave the mover with no king (not needed in
//!     Chaturaji, since there is no check rule – kept as a stub for clarity)

use crate::board::{bit, Board, Move};
use crate::movegen::MoveGen;
use crate::piece::{Color, PieceKind};

/// Stateless rule-checker.
pub struct Rules;

impl Rules {
    // ─── Legal move enumeration ───────────────────────────────────────────────

    /// Returns all fully legal moves for the current player.
    pub fn legal_moves(board: &Board) -> Vec<Move> {
        // In Chaturaji there is no check/checkmate, so all pseudo-legal moves
        // are legal.  We keep this wrapper so the engine always calls Rules,
        // not MoveGen directly, making future rule additions easy.
        MoveGen::generate(board)
    }

    // ─── Move application with full rule enforcement ───────────────────────────

    /// Apply `mv` and return the new board with all rule effects resolved.
    pub fn apply_with_effects(board: &Board, mv: Move) -> Board {
        let mut next = board.apply_move(mv);
        Self::apply_check_bonus(&mut next, mv.mover);
        next
    }

    // ─── Check bonus ──────────────────────────────────────────────────────────
    //
    // After a move, count how many active opponents' kings are attacked
    // by the mover's pieces.
    //   1 king attacked → no bonus (single check)
    //   2 kings attacked → +1 (double check)
    //   3 kings attacked → +5 (triple check)
    //
    // "Attacked" means the mover has at least one legal move to the king's square.

    fn apply_check_bonus(board: &mut Board, mover: Color) {
        let attacked_kings = Self::count_attacked_kings(board, mover);
        let bonus = match attacked_kings {
            2 => 1,
            3 => 5,
            _ => 0,
        };
        if bonus > 0 {
            board.scores.add(mover, bonus);
        }
    }

    /// Count how many *active* opponents' kings the `mover` attacks.
    pub fn count_attacked_kings(board: &Board, mover: Color) -> usize {
        let attacked = Self::attacked_squares(board, mover);
        Color::ALL
            .iter()
            .filter(|&&c| c != mover && board.active[c.idx()])
            .filter(|&&c| board.pieces(c, PieceKind::King) & attacked != 0)
            .count()
    }

    /// Bitboard of all squares the `mover` attacks (used for check detection).
    /// This is a fast approximation: generates pseudo-legal moves and marks destinations.
    pub fn attacked_squares(board: &Board, mover: Color) -> u64 {
        // Temporarily set to_move to `mover` so MoveGen generates their moves.
        let mut tmp = board.clone();
        tmp.to_move = mover;
        MoveGen::generate(&tmp)
            .iter()
            .fold(0u64, |acc, mv| acc | bit(mv.to))
    }

    // ─── Aufgabe und Zeitüberschreitung ───────────────────────────────────────

    /// Der Spieler am Zug gibt auf (oder überschreitet die Zeit).
    ///
    /// Er scheidet aus, seine Figuren bleiben stehen — auch der König, der
    /// danach noch 3 Punkte wert ist. `MoveGen` erzeugt für einen inaktiven
    /// Spieler keine Züge, und die Zugrechtweitergabe überspringt ihn ohnehin;
    /// es genügt also, das Flag zu löschen und weiterzureichen.
    ///
    /// Im Self-Play kommt das nicht vor — dort scheidet man nur durch den
    /// Verlust des Königs aus. Gebraucht wird es beim Einlesen echter Partien.
    pub fn resign(board: &Board) -> Board {
        let mut next = board.clone();
        next.active[board.to_move.idx()] = false;

        let mut next_player = board.to_move.next();
        for _ in 0..4 {
            if next.active[next_player.idx()] { break; }
            next_player = next_player.next();
        }
        next.to_move = next_player;
        next
    }

    // ─── Endstand ─────────────────────────────────────────────────────────────

    /// Der Punktestand am Partieende, einschließlich der Könige, die nie
    /// geschlagen wurden.
    ///
    /// Bleibt genau ein Spieler übrig, bekommt er für jeden noch stehenden
    /// König eines Ausgeschiedenen 3 Punkte. Stehen bleiben können nur Könige
    /// von Spielern, die aufgegeben haben — ein geschlagener König ist vom
    /// Brett und wurde bereits verrechnet.
    ///
    /// An 1000 echten Partien gemessen: ohne diesen Zuschlag stimmten 352
    /// Endstände mit chess.com überein, mit ihm 927.
    pub fn final_scores(board: &Board) -> [i32; 4] {
        let mut scores = board.scores.as_array();

        let survivors: Vec<Color> = Color::ALL
            .into_iter()
            .filter(|c| board.active[c.idx()])
            .collect();

        if let [winner] = survivors[..] {
            let standing_kings = Color::ALL
                .iter()
                .filter(|&&c| !board.active[c.idx()])
                .filter(|&&c| board.pieces(c, PieceKind::King) != 0)
                .count() as i32;
            scores[winner.idx()] += standing_kings * PieceKind::King.capture_value();
        }

        scores
    }

    // ─── Game-over ────────────────────────────────────────────────────────────

    /// Returns true when the game has ended (≤1 active player).
    pub fn is_game_over(board: &Board) -> bool {
        board.is_terminal()
    }

    /// The winning player (sole survivor), or None if still running.
    pub fn winner(board: &Board) -> Option<Color> {
        board.winner()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{sq, Board};
    use crate::piece::Color;

    #[test]
    fn legal_moves_non_empty_at_start() {
        let b = Board::default();
        assert!(!Rules::legal_moves(&b).is_empty());
    }

    /// Aufgeben nimmt den Spieler aus der Zugfolge, lässt seine Figuren aber
    /// stehen. Verschwänden sie, wären die Stellungen beim Einlesen echter
    /// Partien falsch besetzt.
    #[test]
    fn resign_skips_player_but_keeps_pieces() {
        let b = Board::default();
        assert_eq!(b.to_move, Color::Red);

        let after = Rules::resign(&b);
        assert!(!after.active[Color::Red.idx()], "Aufgeber ist ausgeschieden");
        assert_eq!(after.to_move, Color::Blue, "Zugrecht geht weiter");
        assert_eq!(
            after.bb[Color::Red.idx()], b.bb[Color::Red.idx()],
            "die Figuren bleiben auf dem Brett",
        );
        assert!(Rules::legal_moves(&after).iter().all(|m| m.mover == Color::Blue));
    }

    /// Gibt der übernächste Spieler ebenfalls auf, muss das Zugrecht über
    /// beide hinwegspringen.
    #[test]
    fn resign_skips_over_already_eliminated() {
        let mut b = Board::default();
        b.active[Color::Blue.idx()] = false;      // Blau ist schon raus
        let after = Rules::resign(&b);            // Rot gibt auf
        assert_eq!(after.to_move, Color::Yellow);
    }

    /// Der König eines Ausgeschiedenen zählt beim Schlagen weiterhin 3 Punkte.
    #[test]
    fn dead_kings_still_score_when_captured() {
        let mut b = Board::empty();
        let d4 = sq(3, 3);
        let d5 = sq(3, 4);

        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]  = bit(d4);
        b.bb[Color::Blue.idx()][PieceKind::King.idx()] = bit(d5);
        b.bb[Color::Red.idx()][PieceKind::King.idx()]  = bit(sq(0, 0));
        b.active = [true, false, false, false];    // Blau hat aufgegeben
        b.to_move = Color::Red;

        let mv = Rules::legal_moves(&b)
            .into_iter()
            .find(|m| m.from == d4 && m.to == d5)
            .expect("Boot schlägt den stehengebliebenen König");
        let after = Rules::apply_with_effects(&b, mv);

        assert_eq!(after.scores.get(Color::Red), PieceKind::King.capture_value());
    }

    /// Bleibt einer übrig, gehören ihm die nie geschlagenen Könige.
    #[test]
    fn survivor_gets_three_per_standing_king() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(7, 7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0, 7));
        // Grün wurde regulär geschlagen: kein König mehr auf dem Brett.
        b.active = [true, false, false, false];
        b.scores.add(Color::Red, 10);

        let scores = Rules::final_scores(&b);
        assert_eq!(scores[Color::Red.idx()], 10 + 2 * 3, "zwei stehende Könige");
        assert_eq!(scores[Color::Blue.idx()], 0);
    }

    /// Solange mehr als einer aktiv ist, gibt es den Zuschlag nicht — sonst
    /// bekäme eine im Ply-Limit abgebrochene Selbstspiel-Partie Punkte
    /// geschenkt.
    #[test]
    fn no_survivor_bonus_while_game_runs() {
        let mut b = Board::default();
        b.active = [true, true, false, false];
        b.scores.add(Color::Red, 7);
        assert_eq!(Rules::final_scores(&b), b.scores.as_array());
    }

    // ─── Doppel- und Dreifachschach ───────────────────────────────────────────
    //
    // Aufbau für alle drei Fälle: ein rotes Boot zieht von d3 nach d4 (Feld
    // sq(3,3)). Von dort greift es die ganze Datei 3 und den ganzen Rang 3 ab.
    // Wie viele Könige darauf stehen, steuert der jeweilige Test.
    //
    //        Datei 3
    //           │
    //   sq(0,3)─┼─sq(3,3)──sq(7,3)      ← Rang 3
    //           │   ▲
    //           │  Boot zieht von sq(3,2) herauf

    /// Baut die Grundstellung: rotes Boot auf d3, roter König abseits.
    fn check_bonus_board(enemy_kings: &[(Color, u8)]) -> Board {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()] = bit(sq(3, 2));
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0, 0));
        for &(c, square) in enemy_kings {
            b.bb[c.idx()][PieceKind::King.idx()] = bit(square);
        }
        b.to_move = Color::Red;
        b
    }

    /// Führt den Zug d3→d4 aus und gibt Rots Punkte zurück. Der Zug schlägt
    /// nichts, der gesamte Punktezuwachs ist also der Schach-Bonus.
    fn score_after_boat_step(b: &Board) -> i32 {
        let mv = Rules::legal_moves(b)
            .into_iter()
            .find(|m| m.from == sq(3, 2) && m.to == sq(3, 3))
            .expect("Boot muss d3→d4 ziehen können");
        assert!(mv.captured.is_none(), "der Testzug darf nichts schlagen");
        Rules::apply_with_effects(b, mv).scores.get(Color::Red)
    }

    /// Ein einzelnes Schach bringt nichts — die Abgrenzung nach unten.
    #[test]
    fn single_check_gives_no_bonus() {
        let b = check_bonus_board(&[
            (Color::Blue,   sq(3, 7)),   // auf der Datei: angegriffen
            (Color::Yellow, sq(7, 7)),   // abseits
            (Color::Green,  sq(0, 7)),   // abseits
        ]);
        assert_eq!(Rules::count_attacked_kings(&b, Color::Red), 1,
                   "das Boot steht schon vor dem Zug auf der Datei des blauen Königs");
        assert_eq!(score_after_boat_step(&b), 0);
    }

    /// Doppelschach: ein Punkt.
    #[test]
    fn double_check_gives_one_point() {
        let b = check_bonus_board(&[
            (Color::Blue,   sq(3, 7)),   // Datei 3
            (Color::Yellow, sq(7, 3)),   // Rang 3
            (Color::Green,  sq(0, 7)),   // abseits
        ]);
        assert_eq!(score_after_boat_step(&b), 1);
    }

    /// Dreifachschach: fünf Punkte.
    #[test]
    fn triple_check_gives_five_points() {
        let b = check_bonus_board(&[
            (Color::Blue,   sq(3, 7)),   // Datei 3, nördlich
            (Color::Yellow, sq(7, 3)),   // Rang 3, östlich
            (Color::Green,  sq(0, 3)),   // Rang 3, westlich
        ]);
        assert_eq!(score_after_boat_step(&b), 5);
    }

    /// Eine verstellte Linie ist kein Schach. Ohne diese Probe könnte
    /// `attacked_squares` Figuren durch andere hindurch angreifen lassen und
    /// die Tests oben wären trotzdem grün.
    #[test]
    fn blocked_line_is_not_check() {
        let mut b = check_bonus_board(&[
            (Color::Blue,   sq(3, 7)),
            (Color::Yellow, sq(7, 3)),
            (Color::Green,  sq(0, 7)),
        ]);
        // Ein roter Bauer stellt die Datei zwischen Boot und blauem König zu.
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = bit(sq(3, 5));
        assert_eq!(score_after_boat_step(&b), 0, "nur noch Gelb im Angriff = einfaches Schach");
    }

    /// Der Bonus richtet sich nach der Zahl der Könige, nicht nach der Zahl
    /// der angreifenden Figuren: zwei Angreifer auf denselben König bleiben
    /// ein einfaches Schach.
    #[test]
    fn two_attackers_on_one_king_is_not_double_check() {
        let mut b = check_bonus_board(&[
            (Color::Blue,   sq(3, 7)),   // einziger angegriffener König
            (Color::Yellow, sq(7, 0)),   // weit ab von Datei 3, Rang 3 und Rang 7
            (Color::Green,  sq(6, 0)),
        ]);
        // Zweites rotes Boot greift denselben blauen König über Rang 7 an.
        // Nach Süden stößt es auf den eigenen König und kommt nicht weiter.
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()] |= bit(sq(0, 7));
        assert_eq!(score_after_boat_step(&b), 0);
    }

}
