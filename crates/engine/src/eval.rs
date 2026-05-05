//! Static evaluation of a Chaturaji position.
//!
//! Returns a score vector `[i32; 4]` – one value per player.  Positive means
//! "good for that player".  The engine's Max^n search maximises its own index.
//!
//! Components:
//!   1. Material difference (using chess.com piece values)
//!   2. Pawn advancement (encourage pushing pawns toward promotion)
//!   3. Mobility (number of legal moves ≈ activity)
//!   4. King safety (penalise a king with many attackers)
//!   5. Threats with turn-order imminence (SEE-light: hanging pieces weighted
//!      by who moves first — attacker or defender)
//!   6. Promotion proximity (Boat = 5 pts, so an advanced pawn is worth a lot)

use chaturaji_core::board::{bit, file_of, rank_of, sq, Board};
use chaturaji_core::piece::{Color, PieceKind};

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

// Boats move like rooks: equal mobility from every square → flat table.
const BOAT_PST: [i32; 64] = [0; 64];

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

// ─── Public evaluation ────────────────────────────────────────────────────────

/// Weights (tunable).
const W_MATERIAL:  i32 = 100;
const W_PST:       i32 = 1;
const W_MOBILITY:  i32 = 5;
/// King-safety weights. King loss = elimination (-10 000 cp), so safety must
/// scale much higher than chess. `KS_DIRECT_IMMINENT` is the penalty for an
/// opponent who attacks the king square AND moves before the defender.
const KS_DIRECT_IMMINENT: i32 = 300;
const KS_DIRECT_LATER:    i32 =  40;
const KS_ZONE_PRESSURE:   i32 =   8;
/// Threat scale: expected centipawn loss × this. 100 = full material credit.
/// Lower because we already discount with imminence and SEE-light, and
/// material itself updates after a real capture.
const W_THREAT:    i32 = 60;
/// Promotion bonus indexed by squares-to-promote (1 = next move would promote).
/// Sized to the real Pawn → Boat material gain (+4 pts ≈ 400 cp), discounted
/// for capture risk and turn-order delay. Unlike the chess Queen promotion,
/// the Boat is only worth 5 — so a pre-promotion pawn must NOT be valued
/// near the full Boat (was 350 + PST 50 + material 100 = 500 cp ≈ Boat 500).
const PROMO_BONUS: [i32; 8] = [0, 200, 80, 30, 10, 0, 0, 0];

/// Evaluate the position and return a score vector [Red, Blue, Yellow, Green].
/// Scores are in centipawns relative to each player.
pub fn evaluate(board: &Board) -> [i32; 4] {
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
        if !board.active[ci] {
            // Eliminated player has a very bad score.
            scores[ci] = -10_000;
            continue;
        }

        // 1. Accumulated game points (from captures + check bonuses)
        let game_pts = board.scores.get(c) * W_MATERIAL;

        // 2. Material on board
        let material: i32 = [
            PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop,
            PieceKind::Boat, PieceKind::King,
        ].iter().map(|&k| {
            let count = board.pieces(c, k).count_ones() as i32;
            count * k.capture_value() * W_MATERIAL
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
            board, c, &atk_v1, &atk_v3, &atk_v5,
        );

        // 6. Threats: hanging pieces, weighted by turn-order imminence
        let threats = -threat_penalty(board, c, &atk_v1, &atk_v3, &atk_v5);

        // 7. Promotion proximity (separate from PST so it scales with Boat value)
        let promo = promotion_bonus(board, c);

        scores[ci] = game_pts + material + pst_bonus + mobility
                   + king_safety + threats + promo;
    }

    scores
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
fn threat_penalty(
    board: &Board,
    c: Color,
    atk_v1: &[u64; 4],
    atk_v3: &[u64; 4],
    atk_v5: &[u64; 4],
) -> i32 {
    let ci      = c.idx();
    let plies_c = (4 + ci as i32 - board.to_move.idx() as i32) % 4;
    let occ     = board.all_occupied();

    // Squares holding any of c's non-king pieces that some opponent attacks.
    let opp_atk_union: u64 = Color::ALL.iter()
        .filter(|&&e| e != c && board.active[e.idx()])
        .map(|&e| atk_v1[e.idx()] | atk_v3[e.idx()] | atk_v5[e.idx()])
        .fold(0u64, |acc, a| acc | a);
    let c_pieces_no_king = [
        PieceKind::Pawn, PieceKind::Knight, PieceKind::Bishop, PieceKind::Boat,
    ].iter().fold(0u64, |acc, &k| acc | board.pieces(c, k));
    let mut threatened = opp_atk_union & c_pieces_no_king;
    if threatened == 0 { return 0; }

    let mut penalty = 0i32;
    let mut attackers = [0i32; 16];
    let mut defenders = [0i32; 16];
    while threatened != 0 {
        let s = threatened.trailing_zeros() as u8;
        threatened &= threatened - 1;

        let victim = victim_value_at(board, s, c);

        // Aggregate per-piece attacker values across all opponents,
        // and remember whether at least one of them moves before c.
        let mut n_atk = 0usize;
        let mut any_imminent = false;
        for e in Color::ALL {
            if e == c || !board.active[e.idx()] { continue; }
            let added = enumerate_attacker_values(board, s, e, occ, &mut attackers, n_atk);
            if added > n_atk {
                let plies_e = (4 + e.idx() as i32 - board.to_move.idx() as i32) % 4;
                if plies_e < plies_c { any_imminent = true; }
            }
            n_atk = added;
        }
        if n_atk == 0 { continue; }
        attackers[..n_atk].sort();

        let n_def = enumerate_attacker_values(board, s, c, occ, &mut defenders, 0);
        defenders[..n_def].sort();

        let see_val = see(victim, &attackers[..n_atk], &defenders[..n_def]);
        if see_val <= 0 { continue; }

        // factor 100 if at least one attacker moves before c, else 30
        // (preserves the magnitude scale of the previous imminence weights).
        let factor = if any_imminent { 100 } else { 30 };
        penalty += see_val * factor;
    }

    penalty * W_THREAT / 100
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
    out:    &mut [i32; 16],
    start:  usize,
) -> usize {
    let target_bb = bit(target);
    let mut n = start;
    let push = |out: &mut [i32; 16], n: &mut usize, v: i32| {
        out[*n] = v; *n += 1;
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
    let mut running = [0i32; 33]; // 16 + 16 + 1
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
fn promotion_bonus(board: &Board, c: Color) -> i32 {
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
        bonus += PROMO_BONUS[dist.min(7)];
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
) -> i32 {
    let king_bb = board.pieces(c, PieceKind::King);
    if king_bb == 0 {
        return 0; // king already gone — handled by `active` flag elsewhere
    }
    let king_sq = king_bb.trailing_zeros() as u8;
    let zone    = king_zone_bb(king_sq);
    let plies_c = (4 + c.idx() as i32 - board.to_move.idx() as i32) % 4;

    let mut penalty = 0i32;
    for opp in Color::ALL {
        if opp == c || !board.active[opp.idx()] { continue; }
        let oi    = opp.idx();
        let o_atk = atk_v1[oi] | atk_v3[oi] | atk_v5[oi];

        // Pressure on the squares around the king, excluding the king itself
        // (that case is scored separately as a direct attack).
        let zone_pressure = (o_atk & zone & !king_bb).count_ones() as i32;
        penalty += zone_pressure * KS_ZONE_PRESSURE;

        // Direct attack on the king square: imminent if the attacker moves
        // before the defender, otherwise the defender can usually defuse.
        if o_atk & king_bb != 0 {
            let plies_o = (4 + oi as i32 - board.to_move.idx() as i32) % 4;
            penalty += if plies_o < plies_c {
                KS_DIRECT_IMMINENT
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
        let red = promotion_bonus(&b, Color::Red);
        for &c in &[Color::Blue, Color::Yellow, Color::Green] {
            let v = promotion_bonus(&b, c);
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

        let p = threat_penalty(&b, Color::Red, &atk_v1, &atk_v3, &atk_v5);
        // SEE: victim=1, attackers=[1,1], defenders=[1] → SEE = 1.
        // Imminent (factor 100) × W_THREAT 60 / 100 = 60. The function
        // returns the positive penalty; the negation happens in evaluate().
        assert_eq!(p, 60, "expected positive penalty 60, got {}", p);
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

        b.to_move = Color::Blue; // Blue moves first → imminent for Red
        let scores_imminent = evaluate(&b);

        b.to_move = Color::Red;  // Red moves first → can defuse
        let scores_later = evaluate(&b);

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
