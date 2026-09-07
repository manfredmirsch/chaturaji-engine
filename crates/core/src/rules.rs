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

/// Punkte je Überlebendem, wenn die Partie an der Zugobergrenze endet.
const SURVIVOR_BONUS: i32 = 2;

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
        Self::apply_check_bonus(board, &mut next, mv.mover);
        next
    }

    // ─── Check bonus ──────────────────────────────────────────────────────────
    //
    // Ein Zug bringt den Bonus, wenn er zwei (+1) oder drei (+5) fremde Könige
    // NEU bedroht.
    //
    // Neu bedroht heißt: der König wird von einem Feld aus angegriffen, von
    // dem aus er vorher nicht angegriffen wurde. Zwei Dinge sind damit nicht
    // verlangt:
    //
    //   * Es muss nicht dieselbe Figur sein. Ein Abzugsschach zählt so gut wie
    //     ein Angriff der gezogenen Figur.
    //   * Der König muss vorher nicht schachfrei gewesen sein. Stand er schon
    //     im Schach und kommt ein zweiter Angreifer hinzu, ist das eine neue
    //     Bedrohung.
    //
    // Der zweite Punkt ist der Unterschied zwischen "neue Bedrohung" und
    // "neues Schach", und er ist groß: über alle 11.558 Partien aus
    // `game_data/` steigen die exakt getroffenen Endstände von 11.274 auf
    // 11.444.
    //
    // An echten Partien belegt:
    //
    //   Partie 100010111, Zug 23, Rot: `g7-g8++` (Bauer d4-d5) gibt Blau auf
    //     e6 Schach — mehr kann der Bauer nicht, er erreicht nur dieses Feld.
    //     Das zweite Schach kommt aus der Linie, die er freigibt.
    //   Partie 100058474, Zug 28, Grün: `Rk10-e10+` (Boot h7-b7) bedroht Gelb
    //     auf b3 neu, während Rots König auf b1 schon vom Läufer auf f5
    //     angegriffen wird — unverändert derselbe Angriff. Eine neue
    //     Bedrohung, ein `+` in der Notation, kein Punkt.
    //
    // Gemessene Lesarten (exakte Endstände von 11.558):
    //
    //   neu bedrohte Könige               11.444   99,0 %   <- diese Regel
    //   Könige neu im Schach des Ziehers  11.274   97,5 %
    //   die gezogene Figur allein         11.276   97,6 %
    //   neu im Schach, von wem auch immer 10.813   93,6 %
    //   alle Könige im Schach              5.095   44,1 %
    //
    // Der verbleibende Fehler ist einseitig: wir zahlen nur noch zu wenig.

    fn apply_check_bonus(before: &Board, next: &mut Board, mover: Color) {
        let bonus = match Self::newly_threatened_kings(before, next, mover) {
            2 => 1,
            3 => 5,
            _ => 0,
        };
        if bonus > 0 {
            next.scores.add(mover, bonus);
        }
    }

    /// Steht der König von `c` im Schach — von wem auch immer?
    pub fn king_in_check(board: &Board, c: Color) -> bool {
        let king = board.pieces(c, PieceKind::King);
        if king == 0 { return false; }
        Color::ALL.iter().any(|&d| {
            d != c && board.active[d.idx()]
                && Self::attacked_squares(board, d) & king != 0
        })
    }

    /// Wie viele fremde Könige bedroht der Ziehende nach dem Zug **neu**?
    ///
    /// Neu heißt: der König wird von einem Feld aus angegriffen, von dem aus
    /// er vorher nicht angegriffen wurde. Das trifft die gezogene Figur auf
    /// ihrem neuen Feld genauso wie eine Figur, deren Linie der Zug erst
    /// freigelegt hat.
    ///
    /// Der Unterschied zum Zustand des Königs: stand er schon im Schach und
    /// kommt eine zweite Figur hinzu, ist das eine neue Bedrohung, aber kein
    /// neues Schach.
    fn newly_threatened_kings(before: &Board, after: &Board, mover: Color) -> usize {
        // Je Brett ein Durchgang durch die Zuggenerierung, danach nur noch
        // Bitmasken — die Königsfelder ändern sich durch einen fremden Zug
        // nicht.
        let mut tmp_before = before.clone();
        tmp_before.to_move = mover;
        let mut tmp_after = after.clone();
        tmp_after.to_move = mover;
        let moves_before = MoveGen::generate(&tmp_before);
        let moves_after  = MoveGen::generate(&tmp_after);

        let attackers = |moves: &[Move], target: u8| -> u64 {
            moves.iter()
                .filter(|m| m.to == target)
                .fold(0u64, |acc, m| acc | bit(m.from))
        };

        Color::ALL
            .iter()
            .filter(|&&c| c != mover && after.active[c.idx()])
            .filter(|&&c| {
                let king = after.pieces(c, PieceKind::King);
                if king == 0 { return false; }
                let ksq = king.trailing_zeros() as u8;
                attackers(&moves_after, ksq) & !attackers(&moves_before, ksq) != 0
            })
            .count()
    }

    /// Wie viele fremde Könige stehen nach dem Zug im Schach *des Ziehenden*,
    /// die es vorher nicht taten?
    ///
    /// Auf welche Figur das zurückgeht, spielt keine Rolle: ein Abzugsschach
    /// zählt genauso wie ein Angriff der gezogenen Figur. Beide Schachs müssen
    /// aber neu sein — ein schon bestehendes zählt nicht noch einmal.
    fn newly_checked_by_mover(before: &Board, after: &Board, mover: Color) -> usize {
        let att_before = Self::attacked_squares(before, mover);
        let att_after  = Self::attacked_squares(after,  mover);
        Color::ALL
            .iter()
            .filter(|&&c| c != mover && after.active[c.idx()])
            .filter(|&&c| {
                after.pieces(c, PieceKind::King) & att_after != 0
                    && before.pieces(c, PieceKind::King) & att_before == 0
            })
            .count()
    }

    /// Wie viele fremde Könige stehen nach dem Zug im Schach, die es vorher
    /// nicht taten — unabhängig davon, wer sie angreift?
    ///
    /// Nur zum Vergleich behalten: diese Lesart trifft an den echten Partien
    /// deutlich schlechter (93,6 % gegen 97,5 %).
    pub fn newly_checked_kings(before: &Board, after: &Board, mover: Color) -> usize {
        Color::ALL
            .iter()
            .filter(|&&c| c != mover && after.active[c.idx()])
            .filter(|&&c| Self::king_in_check(after, c) && !Self::king_in_check(before, c))
            .count()
    }

    /// Wie viele gegnerische Könige greift die Figur auf `from` allein an?
    ///
    /// Nicht wie viele Könige insgesamt im Schach stehen: gefragt ist das
    /// Doppelschach einer einzelnen Figur.
    pub fn kings_attacked_from(board: &Board, mover: Color, from: u8) -> usize {
        let mut tmp = board.clone();
        tmp.to_move = mover;
        let targets = MoveGen::generate(&tmp)
            .iter()
            .filter(|m| m.from == from)
            .fold(0u64, |acc, m| acc | bit(m.to));

        Color::ALL
            .iter()
            .filter(|&&c| c != mover && board.active[c.idx()])
            .filter(|&&c| board.pieces(c, PieceKind::King) & targets != 0)
            .count()
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

    /// Der Punktestand am Partieende.
    ///
    /// Zwei Zuschläge, die sich gegenseitig ausschließen:
    ///
    /// * **Genau ein Überlebender** — die Partie endete durch Ausscheiden der
    ///   anderen. Er bekommt 3 Punkte für jeden noch stehenden König eines
    ///   Ausgeschiedenen. Stehen bleiben können nur Könige von Spielern, die
    ///   aufgegeben haben oder in die Zeit gelaufen sind; ein geschlagener
    ///   König ist vom Brett und schon verrechnet.
    /// * **Mehrere Überlebende** — dann endete die Partie an der Zugobergrenze,
    ///   denn durch Ausscheiden bliebe nur einer übrig. Jeder Überlebende
    ///   bekommt 2 Punkte.
    ///
    /// Die Zahl der Überlebenden ist hier also nicht bloß ein Kriterium,
    /// sondern verrät den Grund des Endes — deshalb braucht die Funktion
    /// keinen zusätzlichen Parameter dafür.
    ///
    /// Beides an allen 11.558 Partien aus `game_data/` gemessen. Der Anteil
    /// exakt getroffener Endstände:
    ///
    /// | Regel                          | Treffer |
    /// |--------------------------------|---------|
    /// | ohne beide Zuschläge           |  30,6 % |
    /// | nur 3 je König                 |  92,7 % |
    /// | 1 statt 3 je König             |  35,9 % |
    /// | 3 je König + 2 je Überlebendem |  96,5 % |
    pub fn final_scores(board: &Board) -> [i32; 4] {
        let mut scores = board.scores.as_array();

        let survivors: Vec<Color> = Color::ALL
            .into_iter()
            .filter(|c| board.active[c.idx()])
            .collect();

        match survivors[..] {
            [winner] => {
                let standing_kings = Color::ALL
                    .iter()
                    .filter(|&&c| !board.active[c.idx()])
                    .filter(|&&c| board.pieces(c, PieceKind::King) != 0)
                    .count() as i32;
                scores[winner.idx()] += standing_kings * PieceKind::King.capture_value();
            }
            _ => {
                for c in &survivors {
                    scores[c.idx()] += SURVIVOR_BONUS;
                }
            }
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

    /// Endet die Partie an der Zugobergrenze, bekommt jeder Überlebende 2
    /// Punkte. Mehr als ein Überlebender heißt genau das: durch Ausscheiden
    /// bliebe nur einer übrig.
    #[test]
    fn move_limit_end_gives_two_per_survivor() {
        let mut b = Board::default();
        b.active = [true, true, false, false];
        b.scores.add(Color::Red, 7);

        let scores = Rules::final_scores(&b);
        assert_eq!(scores[Color::Red.idx()],   7 + 2);
        assert_eq!(scores[Color::Blue.idx()],  2);
        assert_eq!(scores[Color::Yellow.idx()], 0, "Ausgeschiedene bekommen nichts");
        assert_eq!(scores[Color::Green.idx()],  0);
    }

    /// Die beiden Zuschläge schließen sich aus: bei einem Überlebenden zählen
    /// die Könige, nicht die 2 Punkte.
    #[test]
    fn survivor_bonus_and_king_bonus_are_exclusive() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::King.idx()]  = bit(sq(0, 0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()] = bit(sq(7, 7));
        b.active = [true, false, false, false];

        let scores = Rules::final_scores(&b);
        assert_eq!(scores[Color::Red.idx()], 3, "ein stehender König, keine 2 Punkte obendrauf");
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
    ///
    /// Beide Könige stehen auf Rang 3, den das Boot erst nach dem Zug
    /// bestreicht. Auf Datei 3 dürfen sie nicht stehen — die greift das Boot
    /// schon von d3 aus an, das Schach wäre dann nicht neu.
    #[test]
    fn double_check_gives_one_point() {
        let b = check_bonus_board(&[
            (Color::Blue,   sq(0, 3)),   // Rang 3, westlich
            (Color::Yellow, sq(7, 3)),   // Rang 3, östlich
            (Color::Green,  sq(0, 7)),   // abseits
        ]);
        assert_eq!(score_after_boat_step(&b), 1);
    }

    /// Dreifachschach: fünf Punkte — und zugleich die Probe darauf, dass ein
    /// Abzugsschach mitzählt.
    ///
    /// Das Boot zieht c3→c4 und setzt über Rang 4 Blau und Gelb ins Schach.
    /// Dabei gibt es die lange Diagonale frei, auf der ein Läufer auf a1 steht
    /// — der greift nun Grün auf e5 an. Drei neue Schachs, zwei Figuren.
    #[test]
    fn triple_check_gives_five_points() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]   = bit(sq(2, 2)); // c3
        b.bb[Color::Red.idx()][PieceKind::Bishop.idx()] = bit(sq(0, 0)); // a1
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(7, 0)); // h1
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 3)); // a4
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 3)); // h4
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(4, 4)); // e5
        b.to_move = Color::Red;

        assert_eq!(Rules::count_attacked_kings(&b, Color::Red), 0,
                   "vor dem Zug greift Rot keinen König an");

        let mv = Rules::legal_moves(&b)
            .into_iter()
            .find(|m| m.from == sq(2, 2) && m.to == sq(2, 3))
            .expect("Boot muss c3→c4 ziehen können");
        let after = Rules::apply_with_effects(&b, mv);

        assert_eq!(Rules::count_attacked_kings(&after, Color::Red), 3);
        assert_eq!(after.scores.get(Color::Red), 5);
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

    /// Der Zug muss das Doppelschach *geben*. Bleibt die Lage nur bestehen,
    /// gibt es nichts — sonst kassierte ein Spieler den Punkt in jedem
    /// weiteren Zug erneut, auch für einen Zug am anderen Ende des Bretts.
    #[test]
    fn standing_double_check_is_not_paid_again() {
        let mut b = Board::empty();
        // Boot steht bereits auf d4 und greift zwei Könige an.
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()] = bit(sq(3, 3));
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = bit(sq(0, 0));
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = bit(sq(6, 0));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(3, 7)); // Datei 3
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 3)); // Rang 3
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(0, 6)); // abseits
        b.to_move = Color::Red;

        assert_eq!(Rules::count_attacked_kings(&b, Color::Red), 2,
                   "das Doppelschach steht schon vor dem Zug");

        // Ein Bauernzug am anderen Ende des Bretts ändert daran nichts.
        let mv = Rules::legal_moves(&b)
            .into_iter()
            .find(|m| m.from == sq(6, 0) && m.to == sq(6, 1))
            .expect("Bauer muss ziehen können");
        let after = Rules::apply_with_effects(&b, mv);

        assert_eq!(Rules::count_attacked_kings(&after, Color::Red), 2, "Lage unverändert");
        assert_eq!(after.scores.get(Color::Red), 0, "kein neues Schach, kein Punkt");
    }

    /// Ein zweiter Angreifer auf einen bereits bedrohten König zählt als neue
    /// Bedrohung. Genau hier gehen "neue Bedrohung" und "neues Schach"
    /// auseinander.
    ///
    /// Aufbau: der Läufer auf a1 greift Blaus König auf d4 schon vor dem Zug
    /// an. Das Boot zieht c1→c4 und bedroht von dort beide — Blau zusätzlich
    /// über Reihe 4, Gelb auf a4 zum ersten Mal.
    #[test]
    fn a_second_attacker_counts_as_a_new_threat() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Bishop.idx()] = bit(sq(0, 0)); // a1
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]   = bit(sq(2, 0)); // c1
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(7, 0)); // h1
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(3, 3)); // d4
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0, 3)); // a4
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 7)); // h8
        b.to_move = Color::Red;

        assert_eq!(Rules::count_attacked_kings(&b, Color::Red), 1,
                   "Blau steht schon vor dem Zug im Schach des Läufers");

        let mv = Rules::legal_moves(&b)
            .into_iter()
            .find(|m| m.from == sq(2, 0) && m.to == sq(2, 3))
            .expect("Boot muss c1→c4 ziehen können");
        let after = Rules::apply_with_effects(&b, mv);

        assert_eq!(after.scores.get(Color::Red), 1,
                   "Blau kommt ein Angreifer hinzu, Gelb ist neu bedroht");
    }

    /// Zwei Könige im Schach, aber von zwei verschiedenen Figuren — kein
    /// Punkt. Das ist Partie 100058474, Zug 28: das Boot setzt Gelb ins
    /// Schach, während Rot schon vom Läufer bedroht wird.
    ///
    /// Aufbau: Läufer auf a1 greift über die lange Diagonale Blaus König auf
    /// d4 an, und zwar schon vor dem Zug. Das Boot zieht h8→h6 und greift von
    /// dort über Reihe 6 Gelbs König auf e6 an.
    #[test]
    fn two_kings_checked_by_two_pieces_pays_nothing() {
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Bishop.idx()] = bit(sq(0, 0)); // a1
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]   = bit(sq(7, 7)); // h8
        b.bb[Color::Red.idx()][PieceKind::King.idx()]   = bit(sq(6, 0)); // g1
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(3, 3)); // d4
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(4, 5)); // e6
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(0, 6)); // a7
        b.to_move = Color::Red;

        assert_eq!(Rules::count_attacked_kings(&b, Color::Red), 1,
                   "vor dem Zug hält der Läufer Blau im Schach");

        let mv = Rules::legal_moves(&b)
            .into_iter()
            .find(|m| m.from == sq(7, 7) && m.to == sq(7, 5))
            .expect("Boot muss h8→h6 ziehen können");
        let after = Rules::apply_with_effects(&b, mv);

        assert_eq!(Rules::count_attacked_kings(&after, Color::Red), 2,
                   "danach stehen zwei Könige im Schach");
        assert_eq!(Rules::kings_attacked_from(&after, Color::Red, sq(7, 5)), 1,
                   "das Boot allein greift nur einen an");
        assert_eq!(after.scores.get(Color::Red), 0,
                   "zwei Figuren, zwei Könige — kein Doppelschach");
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
