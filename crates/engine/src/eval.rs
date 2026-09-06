//! Static evaluation of a Chaturaji position.
//!
//! Two layers, and the distinction matters:
//!
//! * [`evaluate_raw`] returns a **point estimate** per player — "how many
//!   chess.com points will this player finish with", × 100.
//! * [`evaluate`] pushes that through [`crate::utility::to_utility`] and
//!   returns **expected placement** instead. This is what the search
//!   maximises, because the game is scored by rank, not by point count.
//!
//! Raw components:
//!   1. Material difference (using chess.com piece values)
//!   2. Pawn advancement (encourage pushing pawns toward promotion)
//!   3. Mobility (number of legal moves ≈ activity)
//!   4. King safety (penalise a king with many attackers)
//!   5. Threats with turn-order imminence (SEE-light: hanging pieces weighted
//!      by who moves first — attacker or defender)
//!   6. Promotion proximity (Boat = 5 pts, so an advanced pawn is worth a lot)

use chaturaji_core::board::{bit, file_of, rank_of, sq, Board};
use chaturaji_core::piece::{Color, PieceKind};

use crate::utility::to_utility;

// ─── Piece-square tables ──────────────────────────────────────────────────────
//
// Flat 64-element tables, indexed by square (a1=0 .. h8=63).
// Used for Red (faces North). Other players' tables are obtained by rotating.

// Rank 6 (one square from promotion) is half the chess value: in Chaturaji
// promotion is to Boat (+4 pts), not Queen (+8 pts), so the "almost-queen"
// bonus must be scaled down. The remaining promotion incentive is carried by
// PROMO_BONUS, which is sized to the real Boat - Pawn delta.
#[rustfmt::skip]
const PAWN_PST: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10,-20,-20, 10, 10,  5,
    5, -5,-10,  0,  0,-10, -5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5,  5, 10, 25, 25, 10,  5,  5,
   10, 10, 20, 30, 30, 20, 10, 10,
   25, 25, 25, 25, 25, 25, 25, 25,
    0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const KNIGHT_PST: [i32; 64] = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP_PST: [i32; 64] = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -20,-10,-10,-10,-10,-10,-10,-20,
];

// Chaturaji has no castling — the chess-style KING_PST that rewards corner
// huddling and punishes centralization does not apply. King safety is handled
// dynamically below; the static table stays neutral.
const KING_PST: [i32; 64] = [0; 64];

// Boats move like rooks. Their mobility is the same from any square, but
// positionally they want to leave the home corner and reach enemy ranks.
// Values increase toward the opponent's back rank (rank 7 in Red's frame).
// All rows are left-right symmetric so the 4-fold start-position test holds.
#[rustfmt::skip]
const BOAT_PST: [i32; 64] = [
    //  a    b    c    d    e    f    g    h
    -5, -5, -5,  0,  0, -5, -5, -5,  // rank 0: home corner, small activation penalty
     0,  0,  0,  5,  5,  0,  0,  0,  // rank 1: leaving home
     0,  0,  5,  5,  5,  5,  0,  0,  // rank 2
     0,  5,  5,  5,  5,  5,  5,  0,  // rank 3
     5,  5,  5, 10, 10,  5,  5,  5,  // rank 4: good central control
     5,  5,  5, 10, 10,  5,  5,  5,  // rank 5
    10, 10, 10, 15, 15, 10, 10, 10,  // rank 6: deep in enemy territory
    10, 10, 15, 15, 15, 15, 10, 10,  // rank 7: enemy back rank
];

// ─── Rotation helpers ──────────────────────────────────────────────────────────

/// Rotate a square 90° clockwise (for Blue's perspective).
#[allow(dead_code)]
fn rot90(sq: u8) -> u8 {
    let f = file_of(sq);
    let r = rank_of(sq);
    // 90° CW: (f,r) → (7-r, f)
    (7 - r) * 8 + f   // ← corrected: new_file = 7-r, new_rank = f  
    // Actually: rot90 of (file=f, rank=r) → (file=r, rank=7-f)
    // sq = rank*8+file → new_sq = (7-f)*8 + r
}

fn rotate_sq(sq: u8, color: Color) -> u8 {
    // Map a square from the player's own perspective to Red's frame, so the
    // Red-oriented PSTs apply uniformly. Each player's "forward" must align
    // with Red's +rank direction:
    //   Red    faces North (+rank) → identity
    //   Blue   faces East  (+file) → rotate 90° CCW so +file becomes +rank
    //   Yellow faces South (-rank) → 180° rotation
    //   Green  faces West  (-file) → rotate 90° CW so -file becomes +rank
    match color {
        Color::Red    => sq,
        Color::Blue   => {
            let f = file_of(sq); let r = rank_of(sq);
            f * 8 + (7 - r)          // 90° CCW: (f,r) → (7-r, f)
        }
        Color::Yellow => 63 - sq,    // 180°:  (f,r) → (7-f, 7-r)
        Color::Green  => {
            let f = file_of(sq); let r = rank_of(sq);
            (7 - f) * 8 + r          // 90° CW:  (f,r) → (r, 7-f)
        }
    }
}

fn pst_value(pst: &[i32; 64], sq: u8, color: Color) -> i32 {
    pst[rotate_sq(sq, color) as usize]
}

// ─── Public helpers ───────────────────────────────────────────────────────────

/// Combined attack bitboard for `color`: every square reachable via any of
/// their pieces (pawn diagonal captures, knight/king leaps, slider rays).
/// Used by the search for piece-safety checks at the root.
pub(crate) fn attack_bitboard(board: &Board, color: Color) -> u64 {
    let occ = board.all_occupied();
    pawn_attack_bb(board.pieces(color, PieceKind::Pawn), color)
        | leaper_attack_bb(board.pieces(color, PieceKind::Knight), &KNIGHT_DELTAS)
        | leaper_attack_bb(board.pieces(color, PieceKind::King),   &KING_DELTAS)
        | slider_attack_bb(board.pieces(color, PieceKind::Bishop), &BISHOP_DIRS, occ)
        | slider_attack_bb(board.pieces(color, PieceKind::Boat),   &BOAT_DIRS,   occ)
}

// ─── Turn order ───────────────────────────────────────────────────────────────

/// How many plies until `c` gets to move: 0 = on move right now, up to 3.
///
/// Eliminated players are skipped, so this reflects the seating that actually
/// applies after knockouts rather than the nominal Red→Blue→Yellow→Green cycle
/// — with one player out, "the opponent after next" is only two plies away,
/// not three. An eliminated `c` never moves again and reports `NEVER_MOVES`.
pub(crate) fn plies_until_move(board: &Board, c: Color) -> i32 {
    if !board.active[c.idx()] {
        return NEVER_MOVES;
    }
    let start = board.to_move.idx();
    let mut steps = 0;
    for k in 0..4 {
        let idx = (start + k) % 4;
        if !board.active[idx] {
            continue;
        }
        if idx == c.idx() {
            return steps;
        }
        steps += 1;
    }
    steps
}

/// Distance reported for a player who will never move again. Finite so that it
/// can be fed to the discount curve without special-casing.
pub(crate) const NEVER_MOVES: i32 = 4;

/// Tempo discount in percent, indexed by plies until the relevant player moves.
///
/// In a four-player game turn order is not a detail: the opponent who moves
/// *next* is a categorically different danger from the one who moves three
/// plies later, because you get one, two or three chances to react in between.
/// This is roughly γ = 0.62 per ply, and it replaces what used to be a handful
/// of scattered "imminent vs. later" constants across threats, king safety,
/// promotion and boat triumph — same idea, one curve.
///
/// Index 4 is `NEVER_MOVES`: an eliminated player poses no future threat at all.
const TEMPO_DISCOUNT: [i32; 5] = [100, 62, 38, 24, 0];

#[inline]
pub(crate) fn tempo_discount(plies: i32) -> i32 {
    TEMPO_DISCOUNT[plies.clamp(0, NEVER_MOVES) as usize]
}

/// Tempo factor honouring [`EvalParams::graded_tempo`]. With grading off every
/// distance counts fully, which is what the evaluation did before the discount
/// curve existed.
#[inline]
fn tempo(plies: i32, graded: bool) -> i32 {
    if graded { tempo_discount(plies) } else { 100 }
}

/// Extra discount applied when the defender is on move before the attacker and
/// therefore gets a chance to defuse the threat first.
const DEFENDER_REPRIEVE: i32 = 30;

// ─── Public evaluation ────────────────────────────────────────────────────────

/// Weights (tunable).
///
/// The raw evaluation is calibrated as **estimated final points × 100**, so
/// `W_MATERIAL = 100` means "one chess.com point". `utility::to_utility`
/// relies on that calibration when it converts to expected placement.
const W_MATERIAL:  i32 = 100;
/// Weight of *on-board* material, as opposed to points already banked in
/// `board.scores`.
///
/// These are fundamentally different quantities and used to share `W_MATERIAL`.
/// At game end on-board material is worth exactly zero — only `scores` decides
/// the standings. A piece on the board is worth something only indirectly: it
/// is future earning potential (it can capture) and defensive substance.
/// Empirically 5 points of material convert to clearly less than 5 points of
/// final score, hence the discount.
const W_BOARD_MATERIAL: i32 = 40;
const W_PST:       i32 = 10;
const W_MOBILITY:  i32 = 5;
/// King-safety weights. Losing the king does **not** zero your score — your
/// banked points survive elimination and still decide your placement (see
/// `evaluate_raw`). What elimination really costs is your remaining *earning
/// potential* plus the 3 points handed to the capturer. At the start a player
/// holds ~20 points of material at `W_BOARD_MATERIAL` ≈ 8 points of potential,
/// so `KS_DIRECT_IMMINENT` is sized to ~12 points rather than the old 50.
const KS_DIRECT_IMMINENT: i32 = 1200;
const KS_DIRECT_LATER:    i32 =  150;
const KS_ZONE_PRESSURE:   i32 =   15;
/// Threat scale: expected centipawn loss × this. 100 = full material credit.
/// Lower because we already discount with imminence and SEE-light, and
/// material itself updates after a real capture.
const W_THREAT:    i32 = 60;
/// Credit side of a threat, paid to the player who will actually make the
/// capture. Larger than `W_THREAT` because the two sides are not the same
/// currency: the capturer *banks* points, which are permanent and decide the
/// standings, while the victim only loses speculative on-board potential.
/// Captures are therefore genuinely positive-sum in points — which is exactly
/// why Chaturaji rewards aggression far more than chess does.
const W_THREAT_GAIN: i32 = 80;
/// Promotion bonus indexed by squares-to-promote (1 = next move would promote).
/// Sized to the real Pawn → Boat material gain (+4 pts ≈ 400 cp), discounted
/// for capture risk and turn-order delay. Unlike the chess Queen promotion,
/// the Boat is only worth 5 — so a pre-promotion pawn must NOT be valued
/// near the full Boat (was 350 + PST 50 + material 100 = 500 cp ≈ Boat 500).
const PROMO_BONUS: [i32; 8] = [0, 200, 80, 30, 10, 0, 0, 0];

/// Coalition weights, only used when [`EvalParams::coalition`] is on.
const W_COALITION:       i32 = 2;
const COALITION_GAP_CAP: i32 = 15;

// ─── Tunable behaviour ────────────────────────────────────────────────────────

/// The evaluation choices that are genuinely open questions, gathered in one
/// place so they can be A/B-tested instead of argued about.
///
/// [`Default`] is the current model. [`EvalParams::legacy`] reproduces the
/// behaviour from before the four-player rework, so a tournament can measure
/// the difference rather than assume it. Every field corresponds to exactly one
/// modelling decision:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvalParams {
    /// Map the point estimate to expected placement (`true`) or hand the raw
    /// point scale to the search (`false`).
    pub rank_transform: bool,
    /// Score for an eliminated player. `None` freezes them at their banked
    /// points — correct under chess.com scoring. `Some(-10_000)` is the old
    /// "elimination is catastrophic" model.
    pub elimination_score: Option<i32>,
    /// Weight of on-board material. Points already banked always count 100.
    pub board_material: i32,
    /// Credit paid to the player who will make a pending capture. `0` reverts
    /// to debiting the victim only, with no idea of who profits.
    pub threat_gain: i32,
    /// Graded tempo discount by turn distance, versus a flat imminent/later split.
    pub graded_tempo: bool,
    /// Penalty for an opponent attacking the king square who moves first.
    pub ks_direct_imminent: i32,
    /// Hand-tuned coalition bonus. Redundant once `rank_transform` is on, and
    /// double-counts if both are enabled — kept switchable to demonstrate that.
    pub coalition: bool,
}

impl Default for EvalParams {
    fn default() -> Self {
        Self {
            rank_transform:     true,
            elimination_score:  None,
            board_material:     W_BOARD_MATERIAL,
            threat_gain:        W_THREAT_GAIN,
            graded_tempo:       true,
            ks_direct_imminent: KS_DIRECT_IMMINENT,
            coalition:          false,
        }
    }
}

impl EvalParams {
    /// The evaluation as it stood before the four-player rework.
    pub fn legacy() -> Self {
        Self {
            rank_transform:     false,
            elimination_score:  Some(-10_000),
            board_material:     W_MATERIAL,
            threat_gain:        0,
            graded_tempo:       false,
            ks_direct_imminent: 5_000,
            coalition:          true,
        }
    }
}

/// Evaluate the position and return the **utility** vector
/// `[Red, Blue, Yellow, Green]` in `UTIL_SCALE` units (≈ ±10 000).
///
/// This is the function the search uses at leaves: it maps the raw point
/// estimate through the rank transform, so what gets maximised is expected
/// *placement*, not expected point count. `Σ evaluate(..) ≈ 0` by construction.
///
/// Use [`evaluate_raw`] when you want the underlying point estimate (analysis
/// output, debugging, weight tuning).
pub fn evaluate(board: &Board) -> [i32; 4] {
    evaluate_with(board, &EvalParams::default())
}

/// [`evaluate`] under an explicit parameter set. Used by the arena to play one
/// configuration against another.
pub fn evaluate_with(board: &Board, p: &EvalParams) -> [i32; 4] {
    let raw = evaluate_raw_with(board, p);
    if p.rank_transform { to_utility(raw, board) } else { raw }
}

/// Raw evaluation: estimated **final points × 100** per player.
///
/// Composed of two conceptually different parts:
///   * *banked* — `board.scores`, already earned and no longer at risk;
///   * *potential* — everything else, an estimate of points still to come.
///
/// An eliminated player keeps the banked part and loses all potential. That is
/// the whole of what elimination means under chess.com scoring: `apply_move`
/// only clears `active`, never `scores`, and the final standings are a pure
/// point ordering. Being eliminated with 22 points still wins the game.
pub fn evaluate_raw(board: &Board) -> [i32; 4] {
    evaluate_raw_with(board, &EvalParams::default())
}

/// [`evaluate_raw`] under an explicit parameter set.
pub fn evaluate_raw_with(board: &Board, p: &EvalParams) -> [i32; 4] {
    let mut scores = [0i32; 4];
    let occ = board.all_occupied();

    // Per-color attack bitboards split by attacker value (1 = pawn, 3 = knight/king,
    // 5 = bishop/boat). Computed once and reused for threats and defenders.
    let mut atk_v1 = [0u64; 4];
    let mut atk_v3 = [0u64; 4];
    let mut atk_v5 = [0u64; 4];
    for c in Color::ALL {
        let ci = c.idx();
        if !board.active[ci] { continue; }
        atk_v1[ci] = pawn_attack_bb(board.pieces(c, PieceKind::Pawn), c);
        atk_v3[ci] = leaper_attack_bb(board.pieces(c, PieceKind::Knight), &KNIGHT_DELTAS)
                  | leaper_attack_bb(board.pieces(c, PieceKind::King),   &KING_DELTAS);
        atk_v5[ci] = slider_attack_bb(board.pieces(c, PieceKind::Bishop), &BISHOP_DIRS, occ)
                  | slider_attack_bb(board.pieces(c, PieceKind::Boat),   &BOAT_DIRS,   occ);
    }

    for c in Color::ALL {
        let ci = c.idx();
        // 1. Accumulated game points (from captures + check bonuses).
        // Banked and safe: elimination cannot take these away.
        let game_pts = board.scores.get(c) * W_MATERIAL;

        if !board.active[ci] {
            // Eliminated: the banked points still count towards the final
            // standings, but no further points can be earned. Freeze here.
            scores[ci] = p.elimination_score.unwrap_or(game_pts);
            continue;
        }

        // 2. Material on board — future earning potential, not points.
        let material: i32 = [
            PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop,
            PieceKind::Boat, PieceKind::King,
        ].iter().map(|&k| {
            let count = board.pieces(c, k).count_ones() as i32;
            count * k.capture_value() * p.board_material
        }).sum();

        // 3. Piece-square table bonus
        let pst_bonus: i32 = pst_for_player(board, c);

        // 4. Mobility — controlled squares, derived from the precomputed
        // attack bitboards (avoid a per-color full move-gen). We count
        // squares attacked that are not blocked by our own pieces, plus
        // single-step pawn pushes (pawns' forward moves aren't included in
        // their attack bitboard, which only carries diagonal captures).
        let mobility = {
            let own     = board.occupied_by(c);
            let attacks = atk_v1[ci] | atk_v3[ci] | atk_v5[ci];
            let pawn_pushes = pawn_push_count(board.pieces(c, PieceKind::Pawn), c, occ);
            ((attacks & !own).count_ones() as i32 + pawn_pushes) * W_MOBILITY
        };

        // 5. King safety. Reuses the precomputed attack bitboards (no extra
        // move-gen). Penalises opponents that attack the king square or its
        // 3×3 zone, and weights direct attacks much higher when the attacker
        // moves before the defender (imminent capture).
        let king_safety = king_safety_penalty(
            board, c, &atk_v1, &atk_v3, &atk_v5, p,
        );

        // 6. Promotion proximity (separate from PST so it scales with Boat value)
        let promo = promotion_bonus(board, c, p);

        // 7. Boat Triumph threat / near-triumph detection
        let triumph = boat_triumph_threat(board, c, p);

        // 8. Pawn structure
        let pawn_struct = pawn_structure_bonus(board, c, occ);

        scores[ci] = game_pts + material + pst_bonus + mobility
                   + king_safety + promo + triumph + pawn_struct;
    }

    // 9. Pending captures, booked as transfers: the owner of a hanging piece
    // loses potential and the opponent who will actually take it gains points.
    // Handled outside the per-player loop because a single threat touches two
    // players at once.
    let transfers = threat_transfers(board, &atk_v1, &atk_v3, &atk_v5, p);
    for ci in 0..4 {
        if board.active[ci] {
            scores[ci] += transfers[ci];
        }
    }

    // 10. Hand-tuned coalition term, off by default. The rank transform already
    // produces this effect exactly — damaging the leader raises every other
    // player's placement probability — so enabling both double-counts it.
    // Retained only so the arena can measure the old model.
    if p.coalition {
        let coalition = coalition_adjustment(board, &atk_v1, &atk_v3, &atk_v5);
        for ci in 0..4 {
            if board.active[ci] {
                scores[ci] += coalition[ci];
            }
        }
    }

    scores
}

/// Bonus for players who are behind and actively pressure the leader's king
/// zone, plus a penalty for the leader when several opponents converge.
///
/// A linear stand-in for the coalition dynamics, superseded by the rank
/// transform. Only reachable via [`EvalParams::coalition`].
fn coalition_adjustment(
    board:  &Board,
    atk_v1: &[u64; 4],
    atk_v3: &[u64; 4],
    atk_v5: &[u64; 4],
) -> [i32; 4] {
    let mut adj = [0i32; 4];
    let leader    = board.scores.leader();
    let leader_ki = leader.idx();

    if !board.active[leader_ki] { return adj; }
    let leader_king_bb = board.pieces(leader, PieceKind::King);
    if leader_king_bb == 0 { return adj; }

    let leader_score = board.scores.get(leader);
    let leader_zone  = king_zone_bb(leader_king_bb.trailing_zeros() as u8);
    let mut n_active_threateners = 0i32;

    for c in Color::ALL {
        let ci = c.idx();
        if c == leader || !board.active[ci] { continue; }

        let gap = (leader_score - board.scores.get(c)).clamp(0, COALITION_GAP_CAP);
        if gap == 0 { continue; }

        let own_atk = atk_v1[ci] | atk_v3[ci] | atk_v5[ci];
        if own_atk & leader_zone != 0 {
            adj[ci] += gap * W_COALITION;
            n_active_threateners += 1;
        }
    }

    if n_active_threateners >= 2 {
        adj[leader_ki] -= n_active_threateners * W_COALITION * 3;
    }

    adj
}

/// Bonus for near-Boat-Triumph patterns, penalty for opponent near-triumphs.
///
/// Boat Triumph (a 2×2 square of 4 boats) is worth 15 game points = ~1500 cp.
/// We reward players who are 1–2 boats away from completing one, and penalise
/// players whose opponent already has 3 boats in a 2×2 pattern.
fn boat_triumph_threat(board: &Board, c: Color, p: &EvalParams) -> i32 {
    let own_boats = board.pieces(c, PieceKind::Boat);
    if own_boats.count_ones() < 2 && {
        // Fast path: also skip if no opponent has ≥ 2 boats.
        let opp_max = Color::ALL.iter()
            .filter(|&&e| e != c && board.active[e.idx()])
            .map(|&e| board.pieces(e, PieceKind::Boat).count_ones())
            .max()
            .unwrap_or(0);
        opp_max < 2
    } { return 0; }

    let mut bonus = 0i32;

    // Own near-triumph patterns. Three boats already in place is a one-move
    // threat, so it is worth what the mover can actually reach: discounted by
    // how long the opponents have to break the pattern up first. Two boats is
    // a multi-round plan and stays undiscounted.
    if own_boats.count_ones() >= 2 {
        let t = tempo(plies_until_move(board, c), p.graded_tempo);
        for rank in 0..7u8 {
            for file in 0..7u8 {
                let mask = bit(sq(file,     rank))
                         | bit(sq(file + 1, rank))
                         | bit(sq(file,     rank + 1))
                         | bit(sq(file + 1, rank + 1));
                match (own_boats & mask).count_ones() {
                    3 => bonus += 800 * t / 100,
                    2 => bonus += 100,
                    _ => {}
                }
            }
        }
    }

    // Opponent 3-in-a-2×2 patterns (immediate triumph threat).
    for opp in Color::ALL {
        let oi = opp.idx();
        if opp == c || !board.active[oi] { continue; }
        let opp_boats = board.pieces(opp, PieceKind::Boat);
        if opp_boats.count_ones() < 3 { continue; }
        // Symmetrically: how alarming an opponent's near-triumph is depends on
        // how soon they get to complete it.
        let opp_t = tempo(plies_until_move(board, opp), p.graded_tempo);
        for rank in 0..7u8 {
            for file in 0..7u8 {
                let mask = bit(sq(file,     rank))
                         | bit(sq(file + 1, rank))
                         | bit(sq(file,     rank + 1))
                         | bit(sq(file + 1, rank + 1));
                if (opp_boats & mask).count_ones() >= 3 {
                    bonus -= 800 * opp_t / 100;
                }
            }
        }
    }

    bonus
}

/// Pawn structure bonuses.
///
/// Connected pawns: +15 cp per pawn that is diagonally supported from behind
/// by another own pawn (they defend each other and are harder to break).
///
/// Free promotion lane: +20 cp per pawn with no other pawn of any color
/// standing between it and its promotion square (on the same file/rank).
/// Capped at distance ≤ 4; stacks with PROMO_BONUS.
fn pawn_structure_bonus(board: &Board, c: Color, occ: u64) -> i32 {
    let own_pawns = board.pieces(c, PieceKind::Pawn);
    if own_pawns == 0 { return 0; }

    // Connected: pawn sits on a square attacked diagonally from behind by
    // another own pawn. `pawn_attack_bb` gives forward-diagonal attacks so a
    // pawn on that set is supported by the attacker behind it.
    let support_squares = pawn_attack_bb(own_pawns, c);
    let connected = (own_pawns & support_squares).count_ones() as i32;

    // Free lane: no pawn (any color) between this pawn and its promotion square.
    let all_pawns: u64 = Color::ALL.iter()
        .fold(0u64, |acc, &e| acc | board.pieces(e, PieceKind::Pawn));
    let _ = occ; // occ available if needed for full blockade checks

    let mut free_bonus = 0i32;
    let mut bb = own_pawns;
    while bb != 0 {
        let s  = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f  = file_of(s);
        let r  = rank_of(s);

        // Build the lane between the pawn and the promotion edge (exclusive of
        // both endpoints so the pawn itself and the promotion square don't count).
        let lane: u64 = match c {
            Color::Red    => (r + 1..7).fold(0, |a, rr| a | bit(sq(f, rr))),
            Color::Blue   => (f + 1..7).fold(0, |a, ff| a | bit(sq(ff, r))),
            Color::Yellow => (1..r)    .fold(0, |a, rr| a | bit(sq(f, rr))),
            Color::Green  => (1..f)    .fold(0, |a, ff| a | bit(sq(ff, r))),
        };
        let dist = match c {
            Color::Red    => (7 - r) as usize,
            Color::Blue   => (7 - f) as usize,
            Color::Yellow => r as usize,
            Color::Green  => f as usize,
        };
        if dist <= 4 && lane & all_pawns == 0 {
            free_bonus += 20;
        }
    }

    connected * 15 + free_bonus
}

fn pst_for_player(board: &Board, c: Color) -> i32 {
    let mut val = 0i32;
    let apply = |bb: u64, pst: &[i32; 64]| -> i32 {
        let mut v = 0; let mut b = bb;
        while b != 0 {
            let sq = b.trailing_zeros() as u8;
            b &= b - 1;
            v += pst_value(pst, sq, c);
        }
        v
    };
    val += apply(board.pieces(c, PieceKind::Pawn),   &PAWN_PST)   * W_PST;
    val += apply(board.pieces(c, PieceKind::Knight), &KNIGHT_PST) * W_PST;
    val += apply(board.pieces(c, PieceKind::Bishop), &BISHOP_PST) * W_PST;
    val += apply(board.pieces(c, PieceKind::Boat),   &BOAT_PST)   * W_PST;
    val += apply(board.pieces(c, PieceKind::King),   &KING_PST)   * W_PST;
    val
}

// ─── Attack bitboards (helpers for threat detection) ─────────────────────────

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1),
];
const KING_DELTAS:   [(i8, i8); 8] = [
    (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1),
];
const BISHOP_DIRS:   [(i8, i8); 4] = [(-1,-1),(-1,1),(1,-1),(1,1)];
const BOAT_DIRS:     [(i8, i8); 4] = [(-1,0),(1,0),(0,-1),(0,1)];

fn pawn_attack_bb(pawns: u64, color: Color) -> u64 {
    let cap_dirs: &[(i8, i8)] = match color {
        Color::Red    => &[(-1, 1),(1, 1)],
        Color::Blue   => &[(1, -1),(1, 1)],
        Color::Yellow => &[(-1,-1),(1,-1)],
        Color::Green  => &[(-1,-1),(-1,1)],
    };
    let mut atk = 0u64;
    let mut bb = pawns;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f = file_of(from) as i8;
        let r = rank_of(from) as i8;
        for &(df, dr) in cap_dirs {
            let nf = f + df; let nr = r + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                atk |= bit(sq(nf as u8, nr as u8));
            }
        }
    }
    atk
}

fn leaper_attack_bb(pieces: u64, deltas: &[(i8, i8); 8]) -> u64 {
    let mut atk = 0u64;
    let mut bb = pieces;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f = file_of(from) as i8;
        let r = rank_of(from) as i8;
        for &(df, dr) in deltas {
            let nf = f + df; let nr = r + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                atk |= bit(sq(nf as u8, nr as u8));
            }
        }
    }
    atk
}

/// Count single-step pawn pushes onto empty squares (no double-step in
/// Chaturaji). Pawns' forward direction depends on their color.
fn pawn_push_count(pawns: u64, color: Color, occ: u64) -> i32 {
    let (df, dr): (i8, i8) = match color {
        Color::Red    => (0,  1),
        Color::Blue   => (1,  0),
        Color::Yellow => (0, -1),
        Color::Green  => (-1, 0),
    };
    let mut count = 0;
    let mut bb = pawns;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let nf = file_of(from) as i8 + df;
        let nr = rank_of(from) as i8 + dr;
        if (0..8).contains(&nf) && (0..8).contains(&nr) {
            let to = sq(nf as u8, nr as u8);
            if occ & bit(to) == 0 {
                count += 1;
            }
        }
    }
    count
}

fn slider_attack_bb(pieces: u64, dirs: &[(i8, i8); 4], occ: u64) -> u64 {
    let mut atk = 0u64;
    let mut bb = pieces;
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let f0 = file_of(from) as i8;
        let r0 = rank_of(from) as i8;
        for &(df, dr) in dirs {
            let (mut f, mut r) = (f0 + df, r0 + dr);
            while (0..8).contains(&f) && (0..8).contains(&r) {
                let s = sq(f as u8, r as u8);
                atk |= bit(s);
                if occ & bit(s) != 0 { break; }
                f += df; r += dr;
            }
        }
    }
    atk
}

// ─── Threats (full SEE + imminence) ──────────────────────────────────────────

/// Penalty for `c`'s pieces that an opponent can profitably capture.
/// Uses static-exchange evaluation: the cheapest opponent attacker initiates
/// the capture sequence, c's defenders recapture (cheapest first), opponents
/// respond with their next-cheapest attacker, etc. Either side may stand-pat
/// at any point — the result is the worst-case material loss for c.
///
/// Imminence: if no attacking opponent moves before `c`, the threat can
/// usually be defused on c's own turn, so the penalty is discounted.
/// Expected point **transfers** from pending captures, one delta per player.
///
/// A capture in Chaturaji is not a symmetric material swing the way it is in
/// chess: it moves a concrete number of points to *one specific* player while
/// the two bystanders gain nothing. Because the game is scored by rank, who
/// collects the points matters as much as who loses the piece — an evaluation
/// that only debits the victim implicitly pays the bystanders exactly as much
/// as the capturer, which is the single biggest distortion in a 4-player
/// position.
///
/// So for every hanging piece we work out both sides:
///   * the owner loses the piece's future potential;
///   * the opponent who *moves first* among the attackers banks the points.
///
/// SEE decides whether the capture is sound at all; the imminence factor
/// discounts threats the defender still gets a move to answer.
///
/// Approximation: the first-moving attacker is credited even when the winning
/// exchange would be started by a cheaper piece belonging to a different
/// opponent. Resolving that exactly needs a per-player SEE, which costs more
/// than the accuracy is worth at this node count.
fn threat_transfers(
    board: &Board,
    atk_v1: &[u64; 4],
    atk_v3: &[u64; 4],
    atk_v5: &[u64; 4],
    p: &EvalParams,
) -> [i32; 4] {
    let mut transfers = [0i32; 4];
    let occ = board.all_occupied();

    let mut attackers = [0i32; 48];
    let mut defenders = [0i32; 48];

    for victim_c in Color::ALL {
        let vi = victim_c.idx();
        if !board.active[vi] { continue; }
        let plies_v = plies_until_move(board, victim_c);

        // Squares holding any of the victim's non-king pieces under attack.
        let opp_atk_union: u64 = Color::ALL.iter()
            .filter(|&&e| e != victim_c && board.active[e.idx()])
            .map(|&e| atk_v1[e.idx()] | atk_v3[e.idx()] | atk_v5[e.idx()])
            .fold(0u64, |acc, a| acc | a);
        let pieces_no_king = [
            PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop, PieceKind::Boat,
        ].iter().fold(0u64, |acc, &k| acc | board.pieces(victim_c, k));
        let mut threatened = opp_atk_union & pieces_no_king;

        while threatened != 0 {
            let s = threatened.trailing_zeros() as u8;
            threatened &= threatened - 1;

            let victim = victim_value_at(board, s, victim_c);

            // Aggregate per-piece attacker values across all opponents, and
            // remember which of them reaches the square earliest in turn order.
            let mut n_atk = 0usize;
            let mut first: Option<(i32, usize)> = None; // (plies until move, idx)
            for e in Color::ALL {
                if e == victim_c || !board.active[e.idx()] { continue; }
                let added = enumerate_attacker_values(board, s, e, occ, &mut attackers, n_atk);
                if added > n_atk {
                    let plies_e = plies_until_move(board, e);
                    if first.map_or(true, |(p, _)| plies_e < p) {
                        first = Some((plies_e, e.idx()));
                    }
                }
                n_atk = added;
            }
            let Some((plies_a, capturer)) = first else { continue };
            attackers[..n_atk].sort();

            let n_def = enumerate_attacker_values(board, s, victim_c, occ, &mut defenders, 0);
            defenders[..n_def].sort();

            let see_val = see(victim, &attackers[..n_atk], &defenders[..n_def]);
            if see_val <= 0 { continue; }

            // How much of the threat is real: discounted by how far away the
            // capturer's move is, and cut again if the defender is on move
            // first and can simply walk away or add a defender.
            let t = tempo(plies_a, p.graded_tempo);
            let factor = if plies_a < plies_v { t } else { t * DEFENDER_REPRIEVE / 100 };
            let magnitude = see_val * factor;

            transfers[vi]       -= magnitude * W_THREAT      / 100;
            transfers[capturer] += magnitude * p.threat_gain / 100;
        }
    }

    transfers
}

fn victim_value_at(board: &Board, s: u8, c: Color) -> i32 {
    let sb = bit(s);
    if board.pieces(c, PieceKind::Pawn)   & sb != 0 { return 1; }
    if board.pieces(c, PieceKind::Knight) & sb != 0 { return 3; }
    if board.pieces(c, PieceKind::Bishop) & sb != 0 { return 5; }
    if board.pieces(c, PieceKind::Boat)   & sb != 0 { return 5; }
    0 // unreachable: caller filters threatened squares to non-king pieces
}

/// Append the per-piece capture-values of every piece of `color` that attacks
/// `target` into `out` starting at index `start`. Returns the new end index.
/// `out` must have capacity for at least 15 more entries (max pieces of one
/// color = 8 pawns + 2 knights + 1 king + 2 bishops + 2 boats = 15).
fn enumerate_attacker_values(
    board:  &Board,
    target: u8,
    color:  Color,
    occ:    u64,
    out:    &mut [i32; 48],
    start:  usize,
) -> usize {
    let target_bb = bit(target);
    let mut n = start;
    let push = |out: &mut [i32; 48], n: &mut usize, v: i32| {
        if *n < out.len() { out[*n] = v; *n += 1; }
    };

    let mut bb = board.pieces(color, PieceKind::Pawn);
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if pawn_attack_bb(bit(from), color) & target_bb != 0 { push(out, &mut n, 1); }
    }
    let mut bb = board.pieces(color, PieceKind::Knight);
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if leaper_attack_bb(bit(from), &KNIGHT_DELTAS) & target_bb != 0 { push(out, &mut n, 3); }
    }
    let mut bb = board.pieces(color, PieceKind::King);
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if leaper_attack_bb(bit(from), &KING_DELTAS) & target_bb != 0 { push(out, &mut n, 3); }
    }
    let mut bb = board.pieces(color, PieceKind::Bishop);
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if slider_attack_bb(bit(from), &BISHOP_DIRS, occ) & target_bb != 0 { push(out, &mut n, 5); }
    }
    let mut bb = board.pieces(color, PieceKind::Boat);
    while bb != 0 {
        let from = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if slider_attack_bb(bit(from), &BOAT_DIRS, occ) & target_bb != 0 { push(out, &mut n, 5); }
    }
    n
}

/// Static exchange evaluation. Returns the optimal net material gain for the
/// attacker side (positive = attacker profits, i.e., loss for defender).
/// `attackers`/`defenders` are sorted ascending lists of piece values.
/// Either side may stand-pat at any stage; the routine picks the optimum.
fn see(victim: i32, attackers: &[i32], defenders: &[i32]) -> i32 {
    if attackers.is_empty() { return 0; }

    // running[d] = cumulative material balance from attacker's perspective
    // *after* the d-th capture. running[0] = 0 (no captures yet).
    // Stack-allocated; the SEE chain can't exceed attackers + defenders + 1.
    let mut running = [0i32; 97]; // 48 + 48 + 1
    let mut last    = 0usize;

    let mut net           = 0i32;
    let mut piece_on_s    = victim;
    let mut a_idx         = 0usize;
    let mut d_idx         = 0usize;
    let mut attacker_turn = true;
    loop {
        let next = if attacker_turn {
            if a_idx >= attackers.len() { break; }
            let v = attackers[a_idx]; a_idx += 1; v
        } else {
            if d_idx >= defenders.len() { break; }
            let v = defenders[d_idx]; d_idx += 1; v
        };
        if attacker_turn { net += piece_on_s; } else { net -= piece_on_s; }
        last += 1;
        running[last] = net;
        piece_on_s    = next;
        attacker_turn = !attacker_turn;
    }

    // Backprop: at each stage the side to move can stop or continue. Attacker
    // (even-d) maximises, defender (odd-d) minimises. Stage 0 is attacker's
    // initial choice — they only initiate if SEE > 0, so we return max(0, …).
    let mut opt = running[last];
    for d in (0..last).rev() {
        let attacker_at_d = (d & 1) == 0;
        opt = if attacker_at_d { running[d].max(opt) } else { running[d].min(opt) };
    }
    opt
}

// ─── Promotion proximity (Punkt 3) ───────────────────────────────────────────

/// Bonus for own pawns close to promotion. Pawn → Boat (5 pts), so a pawn one
/// step from promoting is worth far more than its capture_value of 1.
fn promotion_bonus(board: &Board, c: Color, p: &EvalParams) -> i32 {
    // A pawn that promotes on the very next move only cashes in if its owner
    // actually gets to play it — up to three opponents move in between and any
    // of them can capture or block. Distant pawns are unaffected: their plan
    // spans several rounds anyway, so a one-ply offset is noise.
    let t = tempo(plies_until_move(board, c), p.graded_tempo);

    let mut bonus = 0;
    let mut bb = board.pieces(c, PieceKind::Pawn);
    while bb != 0 {
        let s = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let dist = match c {
            Color::Red    => 7 - rank_of(s),
            Color::Blue   => 7 - file_of(s),
            Color::Yellow => rank_of(s),
            Color::Green  => file_of(s),
        } as usize;
        let raw = PROMO_BONUS[dist.min(7)];
        bonus += if dist <= 1 { raw * t / 100 } else { raw };
    }
    bonus
}

// ─── King safety ─────────────────────────────────────────────────────────────

/// 3×3 bitboard around `sq` (king + 8 neighbours), clipped to the board.
fn king_zone_bb(s: u8) -> u64 {
    let f0 = file_of(s) as i8;
    let r0 = rank_of(s) as i8;
    let mut bb = 0u64;
    for df in -1..=1i8 {
        for dr in -1..=1i8 {
            let nf = f0 + df;
            let nr = r0 + dr;
            if (0..8).contains(&nf) && (0..8).contains(&nr) {
                bb |= bit(sq(nf as u8, nr as u8));
            }
        }
    }
    bb
}

fn king_safety_penalty(
    board: &Board,
    c: Color,
    atk_v1: &[u64; 4],
    atk_v3: &[u64; 4],
    atk_v5: &[u64; 4],
    p: &EvalParams,
) -> i32 {
    let king_bb = board.pieces(c, PieceKind::King);
    if king_bb == 0 {
        return 0; // king already gone — handled by `active` flag elsewhere
    }
    let king_sq = king_bb.trailing_zeros() as u8;
    let zone    = king_zone_bb(king_sq);
    let plies_c = plies_until_move(board, c);

    let mut penalty = 0i32;
    for opp in Color::ALL {
        if opp == c || !board.active[opp.idx()] { continue; }
        let oi      = opp.idx();
        let o_atk   = atk_v1[oi] | atk_v3[oi] | atk_v5[oi];
        let plies_o = plies_until_move(board, opp);
        let t       = tempo(plies_o, p.graded_tempo);

        // Pressure on the squares around the king, excluding the king itself
        // (that case is scored separately as a direct attack). Pressure from an
        // opponent three plies out is worth far less than from the next mover.
        let zone_pressure = (o_atk & zone & !king_bb).count_ones() as i32;
        penalty += zone_pressure * KS_ZONE_PRESSURE * t / 100;

        // Direct attack on the king square: imminent if the attacker moves
        // before the defender, otherwise the defender can usually defuse.
        if o_atk & king_bb != 0 {
            penalty += if plies_o < plies_c {
                p.ks_direct_imminent * t / 100
            } else {
                KS_DIRECT_LATER
            };
        }
    }
    -penalty
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::Board;

    /// In the starting position the four players are rotational mirror images
    /// of each other, so every PST-derived value must be identical. This
    /// guards `rotate_sq` against off-by-quadrant bugs.
    #[test]
    fn pst_is_symmetric_at_start() {
        let b = Board::default();
        let red = pst_for_player(&b, Color::Red);
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            let v = pst_for_player(&b, c);
            assert_eq!(
                v, red,
                "PST for {} should equal Red ({}) at the start, got {}",
                c.name(), red, v
            );
        }
    }

    /// Promoting a Red pawn from rank 6 (one square away) to a Boat on rank
    /// 7 must still be a strict improvement for Red. Before the rebalance,
    /// the pre-promotion pawn was scored equal to the promoted Boat (500 cp
    /// each), so the engine had no incentive to actually promote.
    #[test]
    fn promotion_creates_positive_incentive() {
        use chaturaji_core::board::{bit, sq};
        let mut prepromo = Board::empty();
        // Red king + Red pawn one square from promoting (d7).
        prepromo.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        prepromo.bb[Color::Red.idx()][PieceKind::Pawn.idx()]    = bit(sq(3, 6));
        prepromo.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 7));
        prepromo.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 7));
        prepromo.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 0));

        let mut promoted = prepromo.clone();
        // Replace the pawn with a Boat on rank 7 (d8).
        promoted.bb[Color::Red.idx()][PieceKind::Pawn.idx()] = 0;
        promoted.bb[Color::Red.idx()][PieceKind::Boat.idx()] = bit(sq(3, 7));

        let pre  = evaluate(&prepromo)[Color::Red.idx()];
        let post = evaluate(&promoted)[Color::Red.idx()];
        assert!(post > pre,
            "promoted Boat must score strictly higher than pre-promotion pawn: pre={}, post={}",
            pre, post);
    }

    /// Promotion proximity is also rotationally symmetric: every player has
    /// four pawns at distance 5 (2 pushes from the back rank → 7-2=5? no,
    /// pawns start one step in front of the back rank, distance 6 in our
    /// scheme). What matters is that all four players agree.
    #[test]
    fn promo_bonus_is_symmetric_at_start() {
        let b = Board::default();
        let red = promotion_bonus(&b, Color::Red, &EvalParams::default());
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            let v = promotion_bonus(&b, c, &EvalParams::default());
            assert_eq!(v, red, "promo bonus mismatch for {}", c.name());
        }
    }

    /// `rotate_sq` must map every player's home king square to Red's home
    /// king square (d1 = sq(3,0)). This is the most direct expression of the
    /// invariant: "after rotation, the player should look like Red."
    #[test]
    fn rotate_maps_home_king_to_d1() {
        use chaturaji_core::board::sq;
        let d1 = sq(3, 0);
        assert_eq!(rotate_sq(sq(3, 0), Color::Red),    d1, "Red king d1");
        assert_eq!(rotate_sq(sq(0, 4), Color::Blue),   d1, "Blue king a5");
        assert_eq!(rotate_sq(sq(4, 7), Color::Yellow), d1, "Yellow king e8");
        assert_eq!(rotate_sq(sq(7, 3), Color::Green),  d1, "Green king h4");
    }

    /// Starting position: nobody's king is attacked or near attackers, so
    /// king-safety penalty must be zero for everyone.
    #[test]
    fn king_safety_zero_at_start() {
        let b = Board::default();
        let scores = evaluate(&b);
        // Symmetry: all four scores equal.
        for c in Color::ALL {
            assert_eq!(scores[c.idx()], scores[0],
                "{} score differs from Red at start", c.name());
        }
    }

    /// Bare-king board with the four kings far apart, so no king-safety term
    /// fires and the test observes only the effect under examination.
    fn four_kings_in_corners() -> Board {
        use chaturaji_core::board::{bit, sq};
        let mut b = Board::empty();
        let corners = [sq(0, 0), sq(7, 0), sq(7, 7), sq(0, 7)];
        for (c, &s) in Color::ALL.iter().zip(corners.iter()) {
            b.bb[c.idx()][PieceKind::King.idx()] = bit(s);
        }
        b
    }

    /// Elimination does not erase banked points. A player knocked out with a
    /// big score still finishes ahead of survivors who scored little — the
    /// chess.com standings are a pure point ordering. The old evaluation
    /// hard-coded −10 000 for eliminated players and got this backwards.
    #[test]
    fn eliminated_player_keeps_banked_points() {
        let mut b = four_kings_in_corners();
        // Red is out, but banked 22 points before dying. The rest have 2 each.
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = 0;
        b.active[Color::Red.idx()] = false;
        b.scores.add(Color::Red, 22);
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            b.scores.add(c, 2);
        }

        let u = evaluate(&b);
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            assert!(
                u[Color::Red.idx()] > u[c.idx()],
                "eliminated Red (22 pts) must outrank surviving {} (2 pts): {} vs {}",
                c.name(), u[Color::Red.idx()], u[c.idx()]
            );
        }
    }

    /// The flip side: elimination is still bad when nothing was banked.
    #[test]
    fn eliminated_player_without_points_ranks_last() {
        let mut b = four_kings_in_corners();
        b.bb[Color::Red.idx()][PieceKind::King.idx()] = 0;
        b.active[Color::Red.idx()] = false;

        let u = evaluate(&b);
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            assert!(
                u[Color::Red.idx()] < u[c.idx()],
                "pointless elimination must rank last, {} vs {}",
                u[Color::Red.idx()], u[c.idx()]
            );
        }
    }

    /// The utility layer must preserve the sum invariant on real positions,
    /// not just on synthetic vectors.
    #[test]
    fn evaluate_sums_to_zero_on_start_position() {
        let sum: i32 = evaluate(&Board::default()).iter().sum();
        assert!(sum.abs() <= 4, "Σ evaluate should be ≈ 0, was {sum}");
    }

    // ── Turn order ───────────────────────────────────────────────────────────

    #[test]
    fn plies_until_move_skips_eliminated_players() {
        let mut b = four_kings_in_corners();
        b.to_move = Color::Red;
        assert_eq!(plies_until_move(&b, Color::Red),    0);
        assert_eq!(plies_until_move(&b, Color::Blue),   1);
        assert_eq!(plies_until_move(&b, Color::Yellow), 2);
        assert_eq!(plies_until_move(&b, Color::Green),  3);

        // Knock Blue out: Yellow and Green each move one ply sooner. The old
        // `(4 + idx − to_move) % 4` arithmetic could not express this.
        b.active[Color::Blue.idx()] = false;
        assert_eq!(plies_until_move(&b, Color::Yellow), 1);
        assert_eq!(plies_until_move(&b, Color::Green),  2);
        assert_eq!(plies_until_move(&b, Color::Blue),   NEVER_MOVES);
    }

    #[test]
    fn tempo_discount_decays_and_dies_out() {
        assert_eq!(tempo_discount(0), 100, "the player on move is undiscounted");
        for p in 1..=NEVER_MOVES {
            assert!(
                tempo_discount(p) < tempo_discount(p - 1),
                "discount must fall monotonically, broke at {p}"
            );
        }
        assert_eq!(tempo_discount(NEVER_MOVES), 0, "an eliminated player is no threat");
    }

    /// The same attack must count for less when the attacker is further away
    /// in the turn order — the whole point of a graded discount over the old
    /// binary imminent/later split.
    #[test]
    fn king_attack_fades_with_turn_distance() {
        use chaturaji_core::board::{bit, sq};
        // Green's king is the one on a8 (see `four_kings_in_corners`); a Yellow
        // boat on b8 attacks it along the rank.
        //
        // In *both* variants Yellow still moves before Green, so the attack
        // stays on the "imminent" branch and the test isolates the graded
        // discount itself rather than the old binary imminent/later split:
        // 1200 × 100 % versus 1200 × 38 %.
        let mut b = four_kings_in_corners();
        b.bb[Color::Yellow.idx()][PieceKind::Boat.idx()] = bit(sq(1, 7));

        b.to_move = Color::Yellow; // attacker on move: 0 plies out
        let near = evaluate_raw(&b)[Color::Green.idx()];
        b.to_move = Color::Red;    // attacker 2 plies out, still before Green
        let far = evaluate_raw(&b)[Color::Green.idx()];

        assert!(far > near,
            "an attack two plies out must hurt less than one on move: near={near}, far={far}");
    }

    // ── SEE unit tests ───────────────────────────────────────────────────────

    #[test]
    fn see_unopposed_capture_takes_full_victim() {
        // Pawn captures a boat (5) with no defenders → +5 for attacker.
        assert_eq!(see(5, &[1], &[]), 5);
    }

    #[test]
    fn see_protected_pawn_breaks_even() {
        // Pawn captures pawn, defender pawn recaptures → 0.
        assert_eq!(see(1, &[1], &[1]), 0);
    }

    #[test]
    fn see_knight_takes_protected_pawn_loses_material() {
        // Knight (3) captures pawn (1) defended by pawn (1):
        // continuing loses 1 (1 − 3 + … ≤ 0), so attacker stands pat → SEE = 0.
        assert_eq!(see(1, &[3], &[1]), 0);
    }

    #[test]
    fn see_two_attackers_one_defender_overwhelms() {
        // Two pawns attack a knight defended by a bishop.
        // pawn × knight: +3, bishop × pawn: +3 − 5 = −2, pawn × bishop:
        // −2 + 5 = +3. Attacker can stop at +3 after the first capture
        // because defender's recapture is bad for them. SEE = 3.
        assert_eq!(see(3, &[1, 1], &[5]), 3);
    }

    #[test]
    fn see_defender_refuses_when_recapture_loses_more() {
        // Pawn captures a boat (+5). If the knight defender recaptures the
        // pawn (+5−1 = +4), the attacker's second piece (boat) takes the
        // knight (+4+3 = +7). Defender prefers to stop (running stays at +5)
        // rather than continue (would end at +7). SEE = 5 (boat hangs).
        assert_eq!(see(5, &[1, 5], &[3]), 5);
    }

    // ── Integration: threat_penalty must see overload ────────────────────────

    /// One Red pawn defended by one Red pawn, two Blue pawns attacking it.
    /// The OLD boolean-defender code would have called the victim "defended"
    /// and scored loss = 0. With proper SEE the second Blue attacker
    /// overloads the lone defender and one pawn hangs (SEE = 1).
    #[test]
    fn threat_detects_overloaded_defender() {
        use chaturaji_core::board::{bit, sq};
        let mut b = Board::empty();
        // Red victim on d4, Red defender pawn on e3 (attacks d4 diagonally).
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()]    =
              bit(sq(3, 3))  // d4 = victim
            | bit(sq(4, 2)); // e3 = defender
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        // Two Blue pawn attackers on c3 and c5; Blue captures (df=+1, dr=±1)
        // so both squares hit d4.
        b.bb[Color::Blue.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(2, 2))  // c3
            | bit(sq(2, 4)); // c5
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 0));
        b.to_move = Color::Blue; // imminent for Red

        // Build attack bitboards exactly the way evaluate() does.
        let occ = b.all_occupied();
        let mut atk_v1 = [0u64; 4];
        let mut atk_v3 = [0u64; 4];
        let mut atk_v5 = [0u64; 4];
        for c in Color::ALL {
            let ci = c.idx();
            atk_v1[ci] = pawn_attack_bb(b.pieces(c, PieceKind::Pawn), c);
            atk_v3[ci] = leaper_attack_bb(b.pieces(c, PieceKind::Knight), &KNIGHT_DELTAS)
                       | leaper_attack_bb(b.pieces(c, PieceKind::King),   &KING_DELTAS);
            atk_v5[ci] = slider_attack_bb(b.pieces(c, PieceKind::Bishop), &BISHOP_DIRS, occ)
                       | slider_attack_bb(b.pieces(c, PieceKind::Boat),   &BOAT_DIRS,   occ);
        }

        let t = threat_transfers(&b, &atk_v1, &atk_v3, &atk_v5, &EvalParams::default());
        // Two threats live in this position, and the transfer view books both:
        //
        //   d4: Red pawn attacked by two Blue pawns, defended once.
        //       SEE(victim 1, attackers [1,1], defenders [1]) = 1. Blue is on
        //       move and Red is 3 plies away → imminent, factor 100.
        //       Red −1×100×60/100 = −60, Blue +1×100×80/100 = +80.
        //   c5: Blue pawn attacked by Red's d4 pawn, undefended. SEE = 1, but
        //       Red is 3 plies out *and* Blue is on move first, so the threat
        //       is doubly discounted: tempo 24 % × reprieve 30 % → factor 7.
        //       Blue −4, Red +5.
        //
        // Net: Red −55, Blue +76.
        assert_eq!(t[Color::Red.idx()], -55,
            "expected net −55 for Red, got {}", t[Color::Red.idx()]);
        assert_eq!(t[Color::Blue.idx()], 76,
            "expected net +76 for Blue, got {}", t[Color::Blue.idx()]);
    }

    /// The point of booking threats as transfers: the bystanders must stay at
    /// zero. Under the old victim-only penalty, Yellow and Green profited from
    /// Blue's threat exactly as much as Blue did, which is the core modelling
    /// error in a 4-player position.
    #[test]
    fn threat_credits_only_the_capturer_not_the_bystanders() {
        use chaturaji_core::board::{bit, sq};
        let mut b = Board::empty();
        // Red boat hangs on d4; a single Blue pawn on c3 attacks it undefended.
        b.bb[Color::Red.idx()][PieceKind::Boat.idx()]    = bit(sq(3, 3));
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        b.bb[Color::Blue.idx()][PieceKind::Pawn.idx()]   = bit(sq(2, 2));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 0));
        b.to_move = Color::Blue;

        let occ = b.all_occupied();
        let mut atk_v1 = [0u64; 4];
        let mut atk_v3 = [0u64; 4];
        let mut atk_v5 = [0u64; 4];
        for c in Color::ALL {
            let ci = c.idx();
            atk_v1[ci] = pawn_attack_bb(b.pieces(c, PieceKind::Pawn), c);
            atk_v3[ci] = leaper_attack_bb(b.pieces(c, PieceKind::Knight), &KNIGHT_DELTAS)
                       | leaper_attack_bb(b.pieces(c, PieceKind::King),   &KING_DELTAS);
            atk_v5[ci] = slider_attack_bb(b.pieces(c, PieceKind::Bishop), &BISHOP_DIRS, occ)
                       | slider_attack_bb(b.pieces(c, PieceKind::Boat),   &BOAT_DIRS,   occ);
        }

        let t = threat_transfers(&b, &atk_v1, &atk_v3, &atk_v5, &EvalParams::default());
        assert!(t[Color::Red.idx()]  < 0, "victim Red must be debited");
        assert!(t[Color::Blue.idx()] > 0, "capturer Blue must be credited");
        assert_eq!(t[Color::Yellow.idx()], 0, "bystander Yellow must not profit");
        assert_eq!(t[Color::Green.idx()],  0, "bystander Green must not profit");
    }

    /// When two opponents both attack a hanging piece, the credit goes to the
    /// one who is on move first — they get to take it.
    #[test]
    fn threat_credit_goes_to_the_earlier_mover() {
        use chaturaji_core::board::{bit, sq};
        // Yellow pawn hangs on d5, attacked by a Blue knight (b4) and a Green
        // knight (f6). Knights so that nothing attacks back: the pawn only
        // covers c4/e4, which stay empty, and the kings sit in far corners.
        let build = |to_move: Color| {
            let mut b = Board::empty();
            b.bb[Color::Yellow.idx()][PieceKind::Pawn.idx()]  = bit(sq(3, 4)); // d5
            b.bb[Color::Blue.idx()][PieceKind::Knight.idx()]  = bit(sq(1, 3)); // b4
            b.bb[Color::Green.idx()][PieceKind::Knight.idx()] = bit(sq(5, 5)); // f6
            b.bb[Color::Red.idx()][PieceKind::King.idx()]     = bit(sq(0, 0));
            b.bb[Color::Blue.idx()][PieceKind::King.idx()]    = bit(sq(0, 7));
            b.bb[Color::Yellow.idx()][PieceKind::King.idx()]  = bit(sq(7, 7));
            b.bb[Color::Green.idx()][PieceKind::King.idx()]   = bit(sq(7, 0));
            b.to_move = to_move;
            b
        };
        let transfers_for = |b: &Board| {
            let occ = b.all_occupied();
            let mut atk_v1 = [0u64; 4];
            let mut atk_v3 = [0u64; 4];
            let mut atk_v5 = [0u64; 4];
            for c in Color::ALL {
                let ci = c.idx();
                atk_v1[ci] = pawn_attack_bb(b.pieces(c, PieceKind::Pawn), c);
                atk_v3[ci] = leaper_attack_bb(b.pieces(c, PieceKind::Knight), &KNIGHT_DELTAS)
                           | leaper_attack_bb(b.pieces(c, PieceKind::King),   &KING_DELTAS);
                atk_v5[ci] = slider_attack_bb(b.pieces(c, PieceKind::Bishop), &BISHOP_DIRS, occ)
                           | slider_attack_bb(b.pieces(c, PieceKind::Boat),   &BOAT_DIRS,   occ);
            }
            threat_transfers(b, &atk_v1, &atk_v3, &atk_v5, &EvalParams::default())
        };

        // SEE(victim 1, attackers [3,3], defenders []) = 1, imminent in both
        // cases → debit 60, credit 80.
        //
        // Blue on move → Blue reaches the pawn first (0 plies vs Green's 2).
        let t_blue = transfers_for(&build(Color::Blue));
        assert_eq!(t_blue[Color::Yellow.idx()], -60, "victim Yellow must be debited");
        assert_eq!(t_blue[Color::Blue.idx()],    80, "Blue moves first, must be credited");
        assert_eq!(t_blue[Color::Green.idx()],    0, "Green is later, gets nothing");

        // Green on move → the credit flips to Green.
        let t_green = transfers_for(&build(Color::Green));
        assert_eq!(t_green[Color::Yellow.idx()], -60, "victim Yellow must be debited");
        assert_eq!(t_green[Color::Green.idx()],   80, "Green moves first, must be credited");
        assert_eq!(t_green[Color::Blue.idx()],     0, "Blue is later, gets nothing");
    }

    /// Sanity: with proper SEE the same overloaded victim is *not* treated
    /// as "defended" — the OLD boolean-defender check would have scored 0.
    #[test]
    fn threat_overload_differs_from_boolean_defender() {
        use chaturaji_core::board::{bit, sq};
        let mut b = Board::empty();
        b.bb[Color::Red.idx()][PieceKind::Pawn.idx()]    =
              bit(sq(3, 3)) | bit(sq(4, 2));
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(0, 0));
        b.bb[Color::Blue.idx()][PieceKind::Pawn.idx()]   =
              bit(sq(2, 2)) | bit(sq(2, 4));
        b.bb[Color::Blue.idx()][PieceKind::King.idx()]   = bit(sq(0, 7));
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(7, 7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 0));
        b.to_move = Color::Blue;

        let scores = evaluate(&b);
        // The negated penalty must show up in Red's score: not zero.
        // Compare against the same board minus one of the Blue attackers
        // (only one attacker → defender holds → no penalty).
        let mut b1 = b.clone();
        b1.bb[Color::Blue.idx()][PieceKind::Pawn.idx()] = bit(sq(2, 2));
        let scores_one = evaluate(&b1);
        assert!(
            scores[Color::Red.idx()] < scores_one[Color::Red.idx()],
            "overloaded (two attackers) should score worse for Red than \
             defended (one attacker): two={}, one={}",
            scores[Color::Red.idx()], scores_one[Color::Red.idx()]
        );
    }

    /// Place Red's king with a Blue boat raking it from across the board:
    /// king-safety should fire (penalty < 0). It's also Blue's turn, so the
    /// imminent-attacker branch (KS_DIRECT_IMMINENT, larger penalty) applies.
    #[test]
    fn king_safety_penalises_imminent_attack() {
        use chaturaji_core::board::{bit, sq};
        let mut b = Board::empty();
        // Red king alone in the file, a Blue boat staring at it.
        b.bb[Color::Red.idx()][PieceKind::King.idx()]    = bit(sq(3, 0));
        b.bb[Color::Blue.idx()][PieceKind::Boat.idx()]   = bit(sq(3, 7));
        // The other two players need a king to stay "active".
        b.bb[Color::Yellow.idx()][PieceKind::King.idx()] = bit(sq(0, 7));
        b.bb[Color::Green.idx()][PieceKind::King.idx()]  = bit(sq(7, 0));

        // Compared on the raw scale, because the assertion below is stated in
        // centipoints (`KS_DIRECT_*`). The utility scale is a different unit.
        b.to_move = Color::Blue; // Blue moves first → imminent for Red
        let scores_imminent = evaluate_raw(&b);

        b.to_move = Color::Red;  // Red moves first → can defuse
        let scores_later = evaluate_raw(&b);

        // Red's eval is worse when the attacker moves first.
        assert!(
            scores_imminent[Color::Red.idx()] < scores_later[Color::Red.idx()],
            "imminent: {}, later: {}",
            scores_imminent[Color::Red.idx()],
            scores_later[Color::Red.idx()],
        );
        // And the imminent penalty should reflect at least KS_DIRECT_IMMINENT.
        let delta = scores_later[Color::Red.idx()] - scores_imminent[Color::Red.idx()];
        assert!(delta >= KS_DIRECT_IMMINENT - KS_DIRECT_LATER,
            "delta = {}, expected ≥ {}",
            delta, KS_DIRECT_IMMINENT - KS_DIRECT_LATER);
    }
}
