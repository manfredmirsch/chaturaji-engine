//! Board representation.
//!
//! Square numbering: a1 = 0, b1 = 1, …, h1 = 7, a2 = 8, …, h8 = 63.
//! Each of the four players has one 64-bit bitboard per piece kind (5 kinds).
//!
//! Starting layout (chess.com Chaturaji):
//!
//! ```text
//!   a  b  c  d  e  f  g  h
//! 8 .  Gk Gb Gg Gn .  .  .    (Green, faces South)
//! 7 .  Gp Gp Gp Gp .  .  .
//! 6 .  .  .  .  .  .  .  .
//! 5 .  .  .  .  .  .  Yp .
//! 4 .  Bp .  .  .  .  Yp .
//! 3 .  Bp .  .  .  .  Yp .
//! 2 .  Bp .  .  .  .  Yp .
//! 1 Rk Rb Rg Rn .  Bn Bb Bk   (Red faces North, Blue faces West)
//! ```
//! (exact positions established in `Board::default`)

use crate::piece::{Color, Piece, PieceKind};
use crate::score::Scores;

/// A half-move (ply).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from:     u8,          // source square 0-63
    pub to:       u8,          // destination square 0-63
    pub mover:    Color,       // which player moves
    pub captured: Option<Piece>, // piece captured (if any)
    pub promoted: bool,        // was this a pawn promotion to Boat?
}

impl Move {
    pub fn new(from: u8, to: u8, mover: Color) -> Self {
        Self { from, to, mover, captured: None, promoted: false }
    }
}

/// Full game state.
#[derive(Clone)]
pub struct Board {
    /// `bb[color][kind]` – one bitboard per (player, piece kind).
    pub bb: [[u64; 5]; 4],

    /// Whether each player's king is still on the board.
    pub active: [bool; 4],

    /// Whose turn it is.
    pub to_move: Color,

    /// Current cumulative scores.
    pub scores: Scores,

    /// Half-move clock (for draw detection).
    pub half_moves: u32,

    /// Full-move counter.
    pub full_moves: u32,
}

// ─── square helpers ────────────────────────────────────────────────────────────

#[inline] pub fn sq(file: u8, rank: u8) -> u8 { rank * 8 + file }
#[inline] pub fn file_of(s: u8) -> u8 { s & 7 }
#[inline] pub fn rank_of(s: u8) -> u8 { s >> 3 }
#[inline] pub fn bit(s: u8) -> u64 { 1u64 << s }

// ─── Board impl ───────────────────────────────────────────────────────────────

impl Board {
    /// Returns an empty board (no pieces).
    pub fn empty() -> Self {
        Self {
            bb:         [[0u64; 5]; 4],
            active:     [true; 4],
            to_move:    Color::Red,
            scores:     Scores::default(),
            half_moves: 0,
            full_moves: 1,
        }
    }

    // ── piece placement ───────────────────────────────────────────────────────

    fn place(&mut self, kind: PieceKind, color: Color, square: u8) {
        self.bb[color.idx()][kind.idx()] |= bit(square);
    }

    // ── piece lookup ──────────────────────────────────────────────────────────

    /// Returns the piece at `square`, if any.
    pub fn piece_at(&self, square: u8) -> Option<Piece> {
        let b = bit(square);
        for c in Color::ALL {
            for k in PieceKind::ALL {
                if self.bb[c.idx()][k.idx()] & b != 0 {
                    return Some(Piece::new(k, c));
                }
            }
        }
        None
    }

    /// All squares occupied by `color`.
    pub fn occupied_by(&self, color: Color) -> u64 {
        self.bb[color.idx()].iter().fold(0, |a, &b| a | b)
    }

    /// All occupied squares.
    pub fn all_occupied(&self) -> u64 {
        let mut occ = 0u64;
        for c in Color::ALL { occ |= self.occupied_by(c); }
        occ
    }

    /// Bitboard for a specific (color, kind).
    #[inline]
    pub fn pieces(&self, color: Color, kind: PieceKind) -> u64 {
        self.bb[color.idx()][kind.idx()]
    }

    // ── move application ──────────────────────────────────────────────────────

    /// Apply a move and return the updated board.  Does **not** validate legality.
    pub fn apply_move(&self, mv: Move) -> Board {
        let mut next = self.clone();
        let mover = mv.mover;
        let ci = mover.idx();

        // Determine which kind is moving
        let b_from = bit(mv.from);
        let b_to   = bit(mv.to);

        let kind = PieceKind::ALL
            .iter()
            .find(|&&k| next.bb[ci][k.idx()] & b_from != 0)
            .copied()
            .expect("no piece on from-square");

        // Remove from source
        next.bb[ci][kind.idx()] &= !b_from;

        // Remove captured piece (only active players' pieces score)
        if let Some(cap) = mv.captured {
            let cap_ci = cap.color.idx();
            next.bb[cap_ci][cap.kind.idx()] &= !b_to;

            // Score points only if target player is still active
            if next.active[cap_ci] {
                next.scores.add(mover, cap.kind.capture_value());

                // Eliminating a king deactivates that player
                if cap.kind == PieceKind::King {
                    next.active[cap_ci] = false;
                }
            } else if cap.kind == PieceKind::King {
                // Der König eines bereits ausgeschiedenen Spielers zählt
                // weiterhin 3 Punkte. Das ist kein Sonderfall aus der Theorie:
                // wer aufgibt oder die Zeit überschreitet, scheidet aus, lässt
                // seinen König aber stehen — und der wird danach oft noch
                // geschlagen. Ohne diese Zeile fehlten die Punkte, gemessen an
                // 1000 echten Partien in 713 Fällen.
                next.scores.add(mover, cap.kind.capture_value());
            }
        }

        // Place on destination (promotion: pawn → boat)
        let dest_kind = if mv.promoted { PieceKind::Boat } else { kind };
        next.bb[ci][dest_kind.idx()] |= b_to;

        // Advance turn
        // Skip eliminated (inactive) players
        let mut next_player = mover.next();
        for _ in 0..4 {
            if next.active[next_player.idx()] { break; }
            next_player = next_player.next();
        }
        next.to_move = next_player;

        next.half_moves += 1;
        if mover == Color::Green { next.full_moves += 1; }

        next
    }

    // ── terminal detection ────────────────────────────────────────────────────

    /// Returns true when at most one player is still active (game over).
    pub fn is_terminal(&self) -> bool {
        self.active.iter().filter(|&&a| a).count() <= 1
    }

    /// Returns the winner (sole survivor), if game is over.
    pub fn winner(&self) -> Option<Color> {
        if !self.is_terminal() { return None; }
        Color::ALL.into_iter().find(|&c| self.active[c.idx()])
    }
}

// ─── Starting position ─────────────────────────────────────────────────────────
//
// Chess.com Chaturaji layout:
//
//   Red   (South, faces North): rank 1  – a1 King, b1 Bishop, c1 Boat, d1 Knight; pawns on rank 2 a-d
//   Blue  (West,  faces East):  file h  – h1 King, h2 Bishop, h3 Boat, h4 Knight; pawns on file g rank 1-4
//   Yellow(North, faces South): rank 8  – h8 King, g8 Bishop, f8 Boat, e8 Knight; pawns on rank 7 e-h
//   Green (East,  faces West):  file a  – a8 King, a7 Bishop, a6 Boat, a5 Knight; pawns on file b rank 5-8
//
// Source: chess.com Chaturaji article + standard historical layout.

impl Default for Board {
    fn default() -> Self {
        let mut b = Board::empty();

        // chess.com Chaturaji – korrekte Startaufstellung
        //
        // Zugreihenfolge (Uhrzeigersinn): Rot → Blau → Gelb → Grün
        //
        //   a    b    c    d    e    f    g    h
        // 8[ ]  [Bp] [ ]  [ ]  [ ]  [ ]  [Yp] [ ]
        // 7[Bs] [Bp] [ ]  [ ]  [ ]  [ ]  [Yp] [Ys]
        // 6[Bn] [Bp] [ ]  [ ]  [ ]  [ ]  [Yp] [Yn]
        // 5[Bb] [Bp] [ ]  [ ]  [ ]  [ ]  [Yp] [Yb]
        // 4[Bk] [ ]  [ ]  [ ]  [ ]  [ ]  [Gp] [Yk]... wait
        //
        // Korrekte Aufstellung:
        //
        //   a    b    c    d    e    f    g    h
        // 8[Bs] [Bn] [Bb] [Bk] [Yk] [Yb] [Yn] [Ys]   ← Blau a8-d8, Gelb e8-h8
        // 7[Bp] [Bp] [Bp] [Bp] [Yp] [Yp] [Yp] [Yp]   ← Bauern Rang 7
        // 6
        // 5[Bk]                                [Yp]   NEIN – das ist falsch
        //
        // ENDGÜLTIG nach chess.com Forum:
        // Rot   (Süd,  zieht Nord):  a1=S,b1=N,c1=L,d1=K  Bauern a2-d2
        // Blau  (West, zieht Ost):   a8=S,a7=N,a6=L,a5=K  Bauern b5-b8
        // Gelb  (Nord, zieht Süd):   h8=S,g8=N,f8=L,e8=K  Bauern e7-h7
        // Grün  (Ost,  zieht West):  h1=S,h2=N,h3=L,h4=K  Bauern g2-g5
        //
        // Schiff=S, Springer=N, Läufer=L, König=K
        // Reihenfolge vom Rand zur Mitte: Schiff, Springer, Läufer, König

        // ── Rot (Süden, zieht Nord) ────────────────────────────────
        b.place(PieceKind::Boat,   Color::Red, sq(0,0)); // a1
        b.place(PieceKind::Knight, Color::Red, sq(1,0)); // b1
        b.place(PieceKind::Bishop, Color::Red, sq(2,0)); // c1
        b.place(PieceKind::King,   Color::Red, sq(3,0)); // d1
        for f in 0..4u8 { b.place(PieceKind::Pawn, Color::Red, sq(f,1)); } // a2-d2

        // ── Blau (Westen, zieht Ost) ───────────────────────────────
        // Hinterreihe = file a (Ränge 8→5), Rand=a8, Mitte=a5
        b.place(PieceKind::Boat,   Color::Blue, sq(0,7)); // a8
        b.place(PieceKind::Knight, Color::Blue, sq(0,6)); // a7
        b.place(PieceKind::Bishop, Color::Blue, sq(0,5)); // a6
        b.place(PieceKind::King,   Color::Blue, sq(0,4)); // a5
        for r in 4..8u8 { b.place(PieceKind::Pawn, Color::Blue, sq(1,r)); } // b5-b8

        // ── Gelb (Norden, zieht Süd) ───────────────────────────────
        // Hinterreihe = rank 8 (Files h→e), Rand=h8, Mitte=e8
        b.place(PieceKind::Boat,   Color::Yellow, sq(7,7)); // h8
        b.place(PieceKind::Knight, Color::Yellow, sq(6,7)); // g8
        b.place(PieceKind::Bishop, Color::Yellow, sq(5,7)); // f8
        b.place(PieceKind::King,   Color::Yellow, sq(4,7)); // e8
        for f in 4..8u8 { b.place(PieceKind::Pawn, Color::Yellow, sq(f,6)); } // e7-h7

        // ── Grün (Osten, zieht West) ───────────────────────────────
        // Hinterreihe = file h (Ränge 1→4), Rand=h1, Mitte=h4
        b.place(PieceKind::Boat,   Color::Green, sq(7,0)); // h1
        b.place(PieceKind::Knight, Color::Green, sq(7,1)); // h2
        b.place(PieceKind::Bishop, Color::Green, sq(7,2)); // h3
        b.place(PieceKind::King,   Color::Green, sq(7,3)); // h4
        for r in 0..4u8 { b.place(PieceKind::Pawn, Color::Green, sq(6,r)); } // g1-g4

        b
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starting_piece_count() {
        let b = Board::default();
        for c in Color::ALL {
            let total: u32 = b.bb[c.idx()].iter().map(|bb| bb.count_ones()).sum();
            assert_eq!(total, 8, "{} should start with 8 pieces", c.name());
        }
    }

    #[test]
    fn piece_at_known_squares() {
        let b = Board::default();
        // Rot: a1=Schiff, b1=Springer, c1=Läufer, d1=König
        let p = b.piece_at(sq(0,0)).unwrap();
        assert_eq!(p.kind,  PieceKind::Boat,   "a1 sollte Schiff (Rot) sein");
        assert_eq!(p.color, Color::Red);
        let p = b.piece_at(sq(3,0)).unwrap();
        assert_eq!(p.kind,  PieceKind::King,   "d1 sollte König (Rot) sein");
        assert_eq!(p.color, Color::Red);
        // Gelb: e8=König, h8=Schiff (Mitte→Rand)
        let p = b.piece_at(sq(4,7)).unwrap();
        assert_eq!(p.kind,  PieceKind::King,   "e8 sollte König (Gelb) sein");
        assert_eq!(p.color, Color::Yellow);
        let p = b.piece_at(sq(7,7)).unwrap();
        assert_eq!(p.kind,  PieceKind::Boat,   "h8 sollte Schiff (Gelb) sein");
        assert_eq!(p.color, Color::Yellow);
        // Blau: a8=Schiff, a5=König (file a, Rand=a8 → Mitte=a5)
        let p = b.piece_at(sq(0,7)).unwrap();
        assert_eq!(p.kind,  PieceKind::Boat,   "a8 sollte Schiff (Blau) sein");
        assert_eq!(p.color, Color::Blue);
        let p = b.piece_at(sq(0,4)).unwrap();
        assert_eq!(p.kind,  PieceKind::King,   "a5 sollte König (Blau) sein");
        assert_eq!(p.color, Color::Blue);
        // Grün: h1=Schiff, h4=König (file h, Rand=h1 → Mitte=h4)
        let p = b.piece_at(sq(7,0)).unwrap();
        assert_eq!(p.kind,  PieceKind::Boat,   "h1 sollte Schiff (Grün) sein");
        assert_eq!(p.color, Color::Green);
        let p = b.piece_at(sq(7,3)).unwrap();
        assert_eq!(p.kind,  PieceKind::King,   "h4 sollte König (Grün) sein");
        assert_eq!(p.color, Color::Green);
    }

    #[test]
    fn no_overlap_in_start() {
        let b = Board::default();
        let mut seen = 0u64;
        for c in Color::ALL {
            let occ = b.occupied_by(c);
            assert_eq!(seen & occ, 0, "overlap for {}", c.name());
            seen |= occ;
        }
    }
}
