//! Score tracking for all four players.
//!
//! chess.com Chaturaji point system:
//!   Capture a pawn   → +1
//!   Capture a knight → +3
//!   Capture a king   → +3  (eliminates that player)
//!   Capture a bishop → +5
//!   Capture a boat   → +5
//!   Double check     → +1  (applied by rules layer)
//!   Triple check     → +5  (applied by rules layer)

use crate::piece::Color;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Scores(pub [i32; 4]);

impl Scores {
    #[inline]
    pub fn add(&mut self, player: Color, points: i32) {
        self.0[player.idx()] += points;
    }

    #[inline]
    pub fn get(&self, player: Color) -> i32 {
        self.0[player.idx()]
    }

    /// Returns the score vector as an array (used by engine Max^n).
    pub fn as_array(&self) -> [i32; 4] { self.0 }

    /// The player with the highest score.
    pub fn leader(&self) -> Color {
        let (idx, _) = self.0
            .iter()
            .enumerate()
            .max_by_key(|(_, &v)| v)
            .unwrap();
        Color::ALL[idx]
    }
}
