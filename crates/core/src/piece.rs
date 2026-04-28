/// The four players in turn order (chess.com: Red → Blue → Yellow → Green).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Color {
    Red    = 0,
    Blue   = 1,
    Yellow = 2,
    Green  = 3,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Yellow, Color::Green];

    /// Clockwise next player: Red(S) → Blue(W/top-left) → Yellow(N) → Green(E/right)
    #[inline]
    pub fn next(self) -> Color {
        match self {
            Color::Red    => Color::Blue,
            Color::Blue   => Color::Yellow,
            Color::Yellow => Color::Green,
            Color::Green  => Color::Red,
        }
    }

    /// Index 0-3.
    #[inline]
    pub fn idx(self) -> usize { self as usize }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Color::Red    => "Red",
            Color::Blue   => "Blue",
            Color::Yellow => "Yellow",
            Color::Green  => "Green",
        }
    }
}

/// The five piece kinds in Chaturaji (chess.com variant).
///
/// | Kind   | Moves like         | Points |
/// |--------|--------------------|--------|
/// | Pawn   | chess pawn (no dbl)| 1      |
/// | Knight | chess knight       | 3      |
/// | King   | chess king         | 3      |
/// | Bishop | chess bishop       | 5      |
/// | Boat   | chess rook (ortho) | 5      |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PieceKind {
    Pawn   = 0,
    Knight = 1,
    Bishop = 2,
    Boat   = 3,
    King   = 4,
}

impl PieceKind {
    pub const ALL: [PieceKind; 5] = [
        PieceKind::Pawn,
        PieceKind::Knight,
        PieceKind::Bishop,
        PieceKind::Boat,
        PieceKind::King,
    ];

    /// Point value when captured (chess.com rules).
    #[inline]
    pub fn capture_value(self) -> i32 {
        match self {
            PieceKind::Pawn   => 1,
            PieceKind::Knight => 3,
            PieceKind::King   => 3,
            PieceKind::Bishop => 5,
            PieceKind::Boat   => 5,
        }
    }

    pub fn idx(self) -> usize { self as usize }
}

/// A fully identified piece on the board.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind:  PieceKind,
    pub color: Color,
}

impl Piece {
    #[inline]
    pub fn new(kind: PieceKind, color: Color) -> Self { Self { kind, color } }
}
