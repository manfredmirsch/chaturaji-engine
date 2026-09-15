//! Merkmale eines Zuges und ein daraus gelerntes Zugbewertungsmodell.
//!
//! # Warum
//!
//! Die Zugsortierung entscheidet in einer Beam-Suche mehr als die Bewertung.
//! Bei `beam_width 6` und rund 30 legalen Zügen sieht die Suche vier Fünftel
//! aller Züge nie an; steht der beste nicht unter den ersten sechs, existiert
//! er für sie nicht. Sortiert wurde bisher mit zwei Handheuristiken
//! ([`crate::ordering::score_move`] und `move_priority` im Self-Play), die
//! geraten und nie gemessen wurden.
//!
//! Hier stehen stattdessen Merkmale, deren Gewichte aus 6.366 Partien von
//! Spielern ab 2400 geschätzt werden — rund 760.000 beobachtete Entscheidungen
//! samt der Züge, die jeweils *nicht* gewählt wurden. Gebaut wird das Modell in
//! `chaturaji-trainer` (`move_model.rs`), gelesen wird es hier: dasselbe Muster
//! wie beim Eröffnungsbuch, damit die Engine zur Laufzeit ohne
//! Trainer-Abhängigkeit auskommt.
//!
//! # Kosten
//!
//! [`MoveFeatureContext`] hält alles, was für eine Stellung gilt und nicht je
//! Zug neu berechnet werden muss — vor allem die Angriffskarten der vier
//! Spieler. `Rules::attacked_squares` klont das Brett und erzeugt alle Züge;
//! das je Zug zu tun wäre bei 30 Zügen dreißigmal zu teuer.
//!
//! Für die Frage „ist das Zielfeld gedeckt?" wird die Karte *vor* dem Zug
//! benutzt, abzüglich des geschlagenen Steins. Das ist eine Näherung: sie
//! übersieht Deckungen, die der Zug selbst erst freilegt oder verstellt. Der
//! exakte Weg — Zug ausführen und drei Angriffskarten neu erzeugen — kostet
//! rund das Dreißigfache, und für eine Sortierheuristik ist die Näherung der
//! bessere Handel. Wo sie falsch liegt, lernt das Modell das als Rauschen mit.

use serde::{Deserialize, Serialize};

use chaturaji_core::board::{bit, file_of, rank_of, Board, Move};
use chaturaji_core::movegen::MoveGen;
use chaturaji_core::piece::{Color, PieceKind};
use chaturaji_core::rules::Rules;

/// Anzahl der Merkmale. Reihenfolge und Bedeutung siehe [`FEATURE_NAMES`].
pub const N_FEATURES: usize = 40;

/// Namen in der Reihenfolge des Vektors — für die Gewichtstabelle.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "koenig_aus_schach",     //  0  König war im Schach, ist es nach dem Zug nicht mehr
    "schach_vermeiden",      //  1  Zahl der Angreifer auf den eigenen König sinkt
    "umwandlung",            //  2  Bauer wird zum Boot
    "umwandlung_naeher",     //  3  Bauer verringert den Abstand zur Umwandlungslinie
    "schlag_ungedeckt",      //  4  Schlagwert, Zielfeld danach von niemandem gedeckt
    "schlag_gedeckt",        //  5  Schlagwert, Zielfeld gedeckt
    "koenig_geschlagen",     //  6  gegnerischer König fällt (3 Punkte + Ausscheiden)
    "schach_gegeben",        //  7  eine neue Schach-Bedrohung
    "doppelschach",          //  8  zwei neue Schach-Bedrohungen (Bonusregel)
    "figur_gerettet",        //  9  bedrohte Figur zieht auf ein sicheres Feld
    "figur_eingestellt",     // 10  eigene Figur landet ungedeckt im Angriff
    "zentrum",               // 11  Zielfeld in den inneren 4×4
    "ist_bauer",             // 12  Kontrollgröße: Figurenart
    "ist_koenig",            // 13  Kontrollgröße: Königszug
    "deckt_bedrohte",        // 14  deckt eine eigene, angegriffene und bisher
                             //     ungedeckte Figur
    "laesst_haengen",        // 15  gibt die einzige Deckung einer angegriffenen
                             //     eigenen Figur auf
    // ─── Ab hier: für das Policy-Netz ergänzt ────────────────────────────────
    //
    // Für die Linearform wären das tote Gewichte gewesen: „Springer" allein
    // sagt nichts, „Springer, der ins Zentrum zieht" schon. Erst ein Netz kann
    // solche Wechselwirkungen darstellen — und erst seit das gemessen ist
    // (Top-1 27,2 → 32,7 % bei gleichen Merkmalen), lohnt es, mehr Information
    // hineinzugeben.
    "ist_springer",          // 16  Figurenart, einzeln statt nur Bauer/König
    "ist_laeufer",           // 17
    "ist_boot",              // 18
    "weg_nach_vorn",         // 19  Zielfeld näher an der eigenen Umwandlungslinie
    "weg_zur_seite",         // 20  seitlicher Versatz, aus Sicht des Ziehenden
    "weite",                 // 21  Länge des Zuges in Feldern
    "zielfeld_gedeckt",      // 22  eigene Deckung des Zielfelds (0..1)
    "angreifer_zahl",        // 23  wie viele Gegner das Zielfeld angreifen (0..1)
    "schlaegt_gedeckten",    // 24  Schlag auf eine Figur, die ihre Seite deckt
    "zug_zahl",              // 25  wie viele Züge zur Wahl stehen (Kontext)
    // ─── Drohungen und Vierpersonen-Lage ─────────────────────────────────────
    //
    // Zwei Lücken, die bis 2026-09-15 offen standen. Erstens bildete **kein**
    // Merkmal ab, dass ein Zug eine Drohung *erzeugt* — nur, dass er schlägt
    // oder Schach gibt. Zweitens nutzte keins, dass hier vier Spieler mit
    // verschiedenen Punkteständen sitzen: wen man schlägt und wo man selbst
    // steht, ändert den Wert eines Zuges erheblich.
    "greift_an",             // 26  Wert gegnerischer Figuren, die der Stein von
                             //     seinem Zielfeld aus neu angreift
    "greift_ungedeckt_an",   // 27  davon der Teil, den niemand deckt
    "beweglichkeit",         // 28  Felder, die der Stein vom Zielfeld deckt
    "beweglichkeit_gewinn",  // 29  Differenz zum Ausgangsfeld
    "opfer_rang",            // 30  Platz des Bestohlenen, 1 = Führender
    "eigener_rang",          // 31  Platz des Ziehenden
    "punkte_abstand",        // 32  eigener Rückstand auf den Führenden
    "gegner_zahl",           // 33  wie viele Gegner noch im Spiel sind
    // ─── Klassiker, die noch fehlten ─────────────────────────────────────────
    "see_tausch",            // 34  Tauschbilanz beim Schlagen auf gedecktem Feld
    "spiess",                // 35  der Zug reiht zwei gegnerische Figuren auf
    "koenig_naehe",          // 36  Abstand des Zielfelds zum eigenen König
    "feind_koenig_naehe",    // 37  Abstand zum nächsten gegnerischen König
    "bauer_gedeckt",         // 38  Bauer zieht auf ein von eigenem Bauern
                             //     gedecktes Feld
    "freibauer",             // 39  kein gegnerischer Bauer mehr auf der Bahn
];

/// Größter Schlagwert im Spiel (Bishop und Boat). Normiert die Schlagfelder
/// auf [0, 1], damit kein Merkmal die Skala der anderen sprengt.
const MAX_CAPTURE: f32 = 5.0;

/// Was für eine ganze Stellung gilt — einmal berechnen, für alle Züge nutzen.
pub struct MoveFeatureContext {
    /// Angriffskarte je Spieler, in der Stellung *vor* dem Zug.
    attacked: [u64; 4],
    /// Deckungskarte je Spieler — anders als `attacked` **einschließlich** der
    /// Felder mit eigenen Figuren. Ohne das ist nicht zu erkennen, ob eine
    /// angegriffene Figur gedeckt ist oder hängt; siehe `MoveGen::coverage`.
    covered: [u64; 4],
    /// Angegriffen von irgendeinem anderen als dem Ziehenden.
    by_others: u64,
    /// Was die einzelnen Figuren des Ziehenden decken, je Feld. Gebraucht, um
    /// zu fragen „deckt außer dieser noch jemand?" — die Vereinigung aller
    /// anderen lässt sich daraus je Zug billig bilden.
    mover_pieces: Vec<(u8, PieceKind, u64)>,
    /// Belegung vor dem Zug, für `coverage_from` des ziehenden Steins.
    occ: u64,
    /// Wie viele Gegner den König des Ziehenden angreifen.
    own_king_attackers: u32,
    mover: Color,
}

impl MoveFeatureContext {
    pub fn new(board: &Board) -> Self {
        let mover = board.to_move;
        let attacked: [u64; 4] = std::array::from_fn(|i| {
            let c = Color::ALL[i];
            if board.active[i] { Rules::attacked_squares(board, c) } else { 0 }
        });

        let by_others = Color::ALL.iter()
            .filter(|&&c| c != mover)
            .fold(0u64, |acc, &c| acc | attacked[c.idx()]);

        let covered: [u64; 4] = std::array::from_fn(|i| {
            MoveGen::coverage(board, Color::ALL[i])
        });

        let occ = board.all_occupied();
        let mut mover_pieces = Vec::with_capacity(12);
        for kind in PieceKind::ALL {
            let mut bb = board.pieces(mover, kind);
            while bb != 0 {
                let from = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                mover_pieces.push((from, kind, MoveGen::coverage_from(mover, kind, from, occ)));
            }
        }

        let king = board.pieces(mover, PieceKind::King);
        let own_king_attackers = Color::ALL.iter()
            .filter(|&&c| c != mover && board.active[c.idx()])
            .filter(|&&c| attacked[c.idx()] & king != 0)
            .count() as u32;

        Self { attacked, covered, by_others, mover_pieces, occ, own_king_attackers, mover }
    }

    /// Steht das Zielfeld nach dem Zug unter Beschuss?
    ///
    /// Der Beitrag eines geschlagenen Steins fällt weg — der steht nach dem Zug
    /// nicht mehr auf dem Brett. **Die eigene Seite des Opfers zählt aber mit**,
    /// und zwar über die Deckungs- statt die Angriffskarte: ein Läufer, der
    /// einen eigenen Bauern deckt, kann dorthin nicht ziehen, also steht dieses
    /// Feld nie in seiner Angriffskarte. Bis 2026-09-15 fehlte das, und ein
    /// Schlag auf eine gedeckte Figur galt als frei.
    fn defended_by_others(&self, mv: &Move) -> bool {
        let target = bit(mv.to);
        match mv.captured {
            None => self.by_others & target != 0,
            Some(victim) => Color::ALL.iter()
                .filter(|&&c| c != self.mover)
                .any(|&c| if c == victim.color {
                    // Der geschlagene Stein deckt sich nicht selbst; jede andere
                    // Figur seiner Farbe schon.
                    self.covered[c.idx()] & target != 0
                } else {
                    self.attacked[c.idx()] & target != 0
                }),
        }
    }

    /// Deckt außer der Figur auf `ohne` noch eine eigene Figur das Feld `feld`?
    fn gedeckt_ohne(&self, feld: u64, ohne: u8) -> bool {
        self.mover_pieces.iter()
            .filter(|(sq, _, _)| *sq != ohne)
            .any(|(_, _, cov)| cov & feld != 0)
    }
}

/// Abstand eines Feldes zur Umwandlungslinie des Ziehenden.
///
/// Die Richtung ist je Farbe eine andere — Rot nach Norden, Blau nach Osten,
/// Gelb nach Süden, Grün nach Westen. Das ist die klassische Fehlerquelle bei
/// vier Spielern und deshalb hier an einer Stelle festgehalten.
pub fn promotion_distance(mover: Color, sq: u8) -> u8 {
    match mover {
        Color::Red    => 7 - rank_of(sq),
        Color::Blue   => 7 - file_of(sq),
        Color::Yellow => rank_of(sq),
        Color::Green  => file_of(sq),
    }
}

/// Merkmale, die ohne die Stellung *nach* dem Zug auskommen.
///
/// Alles hier ist ein paar Bit-Tests auf den Karten aus dem Kontext. Die
/// teuren Merkmale — alles, was eine neue Angriffskarte braucht — stehen
/// getrennt, weil die Suche sie sich nicht leisten kann (siehe
/// [`fast_features`]).
fn features_cheap(board: &Board, mv: &Move, ctx: &MoveFeatureContext, f: &mut [f32; N_FEATURES]) {
    let mover = board.to_move;
    let moving_kind = board.piece_at(mv.from).map(|p| p.kind);

    // ─── Umwandlung ──────────────────────────────────────────────────────────
    if mv.promoted {
        f[2] = 1.0;
    }
    if moving_kind == Some(PieceKind::Pawn) && !mv.promoted {
        let vorher  = promotion_distance(mover, mv.from);
        let nachher = promotion_distance(mover, mv.to);
        if nachher < vorher {
            // Je näher an der Linie, desto mehr zählt der Schritt.
            f[3] = (7 - nachher) as f32 / 7.0;
        }
    }

    // ─── Schlagen ────────────────────────────────────────────────────────────
    if let Some(victim) = mv.captured {
        let wert = victim.kind.capture_value() as f32 / MAX_CAPTURE;
        if ctx.defended_by_others(mv) { f[5] = wert; } else { f[4] = wert; }
        if victim.kind == PieceKind::King { f[6] = 1.0; }
    }

    // ─── eigene Figuren in Sicherheit / in Gefahr ────────────────────────────
    if let Some(kind) = moving_kind {
        let wert = kind.capture_value() as f32 / MAX_CAPTURE;
        let stand_im_angriff = ctx.by_others & bit(mv.from) != 0;
        let landet_im_angriff = ctx.defended_by_others(mv);
        // „Eingestellt" heißt angegriffen **und** ungedeckt. Dass die eigene
        // Deckung dabei mitzählt, ging bis 2026-09-15 unter — ein Zug auf ein
        // angegriffenes, aber gut gedecktes Feld sah aus wie ein Fehler.
        let eigene_deckung = ctx.gedeckt_ohne(bit(mv.to), mv.from);
        if stand_im_angriff && !landet_im_angriff { f[9]  = wert; }
        if landet_im_angriff && !eigene_deckung && mv.captured.is_none() { f[10] = wert; }

        f[12] = (kind == PieceKind::Pawn) as u8 as f32;
        f[13] = (kind == PieceKind::King) as u8 as f32;
    }

    // ─── Deckung eigener Figuren ─────────────────────────────────────────────
    //
    // Was die vorhandenen Merkmale 9 und 10 nicht erfassen: sie sehen nur den
    // ziehenden Stein. Eine Figur, die stehen bleibt und durch den Zug ihre
    // Deckung verliert, kam darin nicht vor — und genau das ist der Fall, über
    // den sich in der Praxis geärgert wird.
    //
    // Näherung: gerechnet wird mit der Belegung *vor* dem Zug. Dass der
    // ziehende Stein sein Ausgangsfeld räumt und damit Strahlen anderer Figuren
    // öffnet, bleibt unberücksichtigt. Das exakt zu machen kostete eine neue
    // Deckungskarte je Zug; für ein Sortiermerkmal ist das zu teuer.
    if let Some(kind) = moving_kind {
        let von_to = MoveGen::coverage_from(mover, kind, mv.to, ctx.occ);
        let von_from = ctx.mover_pieces.iter()
            .find(|(sq, _, _)| *sq == mv.from)
            .map(|(_, _, cov)| *cov)
            .unwrap_or(0);

        for (feld, art, _) in &ctx.mover_pieces {
            if *feld == mv.from { continue; }          // zieht ja gerade weg
            let b = bit(*feld);
            if ctx.by_others & b == 0 { continue; }    // nicht angegriffen, egal
            let wert = art.capture_value() as f32 / MAX_CAPTURE;
            let sonst_gedeckt = ctx.gedeckt_ohne(b, mv.from);

            if !sonst_gedeckt && von_to & b != 0 && von_from & b == 0 {
                f[14] += wert;   // neu gedeckt
            }
            if !sonst_gedeckt && von_from & b != 0 && von_to & b == 0 {
                f[15] += wert;   // einzige Deckung aufgegeben
            }
        }
    }

    // ─── Ergänzungen für das Policy-Netz ─────────────────────────────────────
    if let Some(kind) = moving_kind {
        f[16] = (kind == PieceKind::Knight) as u8 as f32;
        f[17] = (kind == PieceKind::Bishop) as u8 as f32;
        f[18] = (kind == PieceKind::Boat)   as u8 as f32;

        // Richtung aus Sicht des Ziehenden: „vorwärts" ist für Rot Norden, für
        // Grün Westen. Ohne diese Drehung wären die vier Farben vier
        // verschiedene Aufgaben, und das Netz müsste jede einzeln lernen.
        let vor_von = promotion_distance(mover, mv.from) as f32;
        let vor_nach = promotion_distance(mover, mv.to) as f32;
        f[19] = (vor_von - vor_nach) / 7.0;

        let seite = |sq: u8| -> f32 {
            match mover {
                Color::Red | Color::Yellow => file_of(sq) as f32,
                Color::Blue | Color::Green => rank_of(sq) as f32,
            }
        };
        f[20] = (seite(mv.to) - seite(mv.from)).abs() / 7.0;

        let dr = (rank_of(mv.to) as i32 - rank_of(mv.from) as i32).abs();
        let df = (file_of(mv.to) as i32 - file_of(mv.from) as i32).abs();
        f[21] = dr.max(df) as f32 / 7.0;
    }

    // Wie gut das Zielfeld gedeckt ist und von wie vielen angegriffen — die
    // Aufteilung in „gedeckt/ungedeckt" war bisher binär, obwohl der
    // Unterschied zwischen einem und drei Angreifern erheblich ist.
    let ziel = bit(mv.to);
    f[22] = ctx.gedeckt_ohne(ziel, mv.from) as u8 as f32;
    f[23] = Color::ALL.iter()
        .filter(|&&c| c != mover)
        .filter(|&&c| ctx.attacked[c.idx()] & ziel != 0)
        .count() as f32 / 3.0;

    if let Some(victim) = mv.captured {
        // Eine Figur, die selbst etwas deckt, zu schlagen hat eine Folge über
        // den Materialwert hinaus: was sie deckte, hängt danach.
        let deckt_etwas = ctx.covered[victim.color.idx()] & ziel != 0;
        if deckt_etwas { f[24] = victim.kind.capture_value() as f32 / MAX_CAPTURE; }
    }

    // ─── Drohungen ───────────────────────────────────────────────────────────
    //
    // Was der ziehende Stein von seinem Zielfeld aus bedroht, abzüglich dessen,
    // was er schon vom Ausgangsfeld aus bedrohte. Nur der Zuwachs zählt: eine
    // Figur, die ohnehin schon angegriffen war, ist keine neue Drohung.
    if let Some(kind) = moving_kind {
        let von_to = MoveGen::coverage_from(mover, kind, mv.to, ctx.occ);
        let von_from = ctx.mover_pieces.iter()
            .find(|(sq, _, _)| *sq == mv.from)
            .map(|(_, _, cov)| *cov)
            .unwrap_or(0);
        let neu = von_to & !von_from;

        let mut wert = 0.0f32;
        let mut wert_ungedeckt = 0.0f32;
        for c in Color::ALL {
            if c == mover || !board.active[c.idx()] { continue; }
            let mut bb = board.occupied_by(c) & neu;
            while bb != 0 {
                let sq = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                if sq == mv.to { continue; }   // das ist der Schlag, schon erfasst
                let Some(p) = board.piece_at(sq) else { continue };
                let w = p.kind.capture_value() as f32 / MAX_CAPTURE;
                wert += w;
                // Ungedeckt heißt: die eigene Seite deckt es nicht, und auch
                // kein Dritter. Eine solche Figur ist wirklich in Gefahr.
                let gedeckt = Color::ALL.iter()
                    .filter(|&&d| d != mover)
                    .any(|&d| ctx.covered[d.idx()] & bit(sq) != 0);
                if !gedeckt { wert_ungedeckt += w; }
            }
        }
        f[26] = wert.min(3.0) / 3.0;
        f[27] = wert_ungedeckt.min(3.0) / 3.0;

        // Beweglichkeit: wie viele Felder der Stein danach deckt. Eigene
        // Figuren zählen mit, sie sind ja gedeckt und nicht blockiert im Sinne
        // der Wirkung.
        let bew_to = von_to.count_ones() as f32;
        let bew_from = von_from.count_ones() as f32;
        f[28] = (bew_to / 27.0).min(1.0);           // 27 = Läufer im Zentrum
        f[29] = ((bew_to - bew_from) / 27.0).clamp(-1.0, 1.0);
    }

    // ─── Vierpersonen-Lage ───────────────────────────────────────────────────
    //
    // Wen man schlägt, ist nicht gleichgültig: dem Führenden etwas wegzunehmen
    // hilft der eigenen Platzierung mehr, als den Letzten weiter zu schwächen.
    // Und wer selbst führt, spielt anders als wer hinten liegt.
    let punkte = board.scores.as_array();
    let rang = |c: usize| -> f32 {
        // 0 = Führender, 1 = Letzter. Punktgleiche teilen sich den Rang.
        let besser = (0..4).filter(|&j| board.active[j] && punkte[j] > punkte[c]).count() as f32;
        let aktive = (0..4).filter(|&j| board.active[j]).count().max(1) as f32;
        besser / (aktive - 1.0).max(1.0)
    };
    if let Some(victim) = mv.captured {
        f[30] = 1.0 - rang(victim.color.idx());   // 1 = dem Führenden genommen
    }
    f[31] = 1.0 - rang(mover.idx());
    let spitze = (0..4).filter(|&j| board.active[j]).map(|j| punkte[j]).max().unwrap_or(0);
    f[32] = ((spitze - punkte[mover.idx()]) as f32 / 20.0).clamp(0.0, 1.0);
    f[33] = (0..4).filter(|&j| board.active[j] && j != mover.idx()).count() as f32 / 3.0;

    // ─── Tauschbilanz ────────────────────────────────────────────────────────
    //
    // Bisher unterschied das Modell nur, *ob* das Schlagfeld gedeckt ist. Ob
    // ein Bauer einen Läufer schlägt oder umgekehrt, macht aber den ganzen
    // Unterschied — und genau das ist die Größe, die eine Suche an dieser
    // Stelle statisch abschätzt.
    if let (Some(victim), Some(kind)) = (mv.captured, moving_kind) {
        if ctx.defended_by_others(mv) {
            let bilanz = victim.kind.capture_value() as f32 - kind.capture_value() as f32;
            f[34] = (bilanz / MAX_CAPTURE).clamp(-1.0, 1.0);
        } else {
            // Ungedeckt: der volle Wert bleibt.
            f[34] = victim.kind.capture_value() as f32 / MAX_CAPTURE;
        }
    }

    // ─── Spieß ───────────────────────────────────────────────────────────────
    //
    // Ein Schieber, der von seinem Zielfeld aus zwei gegnerische Figuren auf
    // einer Linie aufreiht: die vordere kann ziehen, die hintere steht dann
    // offen. Nur für Läufer und Boot; Springer und Bauern können das nicht.
    if let Some(kind) = moving_kind {
        let richtungen: &[(i8, i8)] = match kind {
            PieceKind::Bishop => &[(-1,-1),(-1,1),(1,-1),(1,1)],
            PieceKind::Boat   => &[(-1,0),(1,0),(0,-1),(0,1)],
            _ => &[],
        };
        let mut gewinn = 0.0f32;
        for &(df, dr) in richtungen {
            let (mut ff, mut rr) = (file_of(mv.to) as i8 + df, rank_of(mv.to) as i8 + dr);
            let mut erste: Option<u8> = None;
            while (0..8).contains(&ff) && (0..8).contains(&rr) {
                let sq = (rr as u8) * 8 + ff as u8;
                if sq == mv.from { ff += df; rr += dr; continue; }   // zieht ja weg
                if let Some(p) = board.piece_at(sq) {
                    if p.color == mover { break; }                   // eigene Figur blockt
                    match erste {
                        None => erste = Some(p.kind.capture_value() as u8),
                        Some(v1) => {
                            // Der Gewinn ist die kleinere der beiden Figuren:
                            // so viel lässt sich mindestens holen.
                            let v2 = p.kind.capture_value() as u8;
                            gewinn += v1.min(v2) as f32;
                            break;
                        }
                    }
                }
                ff += df; rr += dr;
            }
        }
        f[35] = (gewinn / MAX_CAPTURE).min(1.0);
    }

    // ─── Königsabstände ──────────────────────────────────────────────────────
    let abstand = |a: u8, b: u8| -> f32 {
        let (df, dr) = ((file_of(a) as i32 - file_of(b) as i32).abs(),
                        (rank_of(a) as i32 - rank_of(b) as i32).abs());
        df.max(dr) as f32 / 7.0
    };
    let eigener_koenig = board.pieces(mover, PieceKind::King);
    if eigener_koenig != 0 {
        f[36] = 1.0 - abstand(mv.to, eigener_koenig.trailing_zeros() as u8);
    }
    let naechster = Color::ALL.iter()
        .filter(|&&c| c != mover && board.active[c.idx()])
        .filter_map(|&c| {
            let k = board.pieces(c, PieceKind::King);
            (k != 0).then(|| abstand(mv.to, k.trailing_zeros() as u8))
        })
        .fold(f32::INFINITY, f32::min);
    if naechster.is_finite() { f[37] = 1.0 - naechster; }

    // ─── Bauern ──────────────────────────────────────────────────────────────
    if moving_kind == Some(PieceKind::Pawn) {
        // Von welchen Feldern aus ein eigener Bauer das Zielfeld deckt: das
        // sind genau die, von denen aus er dorthin schlagen könnte.
        let eigene_bauern = board.pieces(mover, PieceKind::Pawn) & !bit(mv.from);
        let (f0, r0) = (file_of(mv.to) as i8, rank_of(mv.to) as i8);
        let rueck: &[(i8, i8)] = match mover {
            Color::Red    => &[(-1,-1),(1,-1)],
            Color::Blue   => &[(-1,-1),(-1,1)],
            Color::Yellow => &[(-1,1),(1,1)],
            Color::Green  => &[(1,-1),(1,1)],
        };
        for &(df, dr) in rueck {
            let (ff, rr) = (f0 + df, r0 + dr);
            if (0..8).contains(&ff) && (0..8).contains(&rr)
                && eigene_bauern & bit((rr as u8) * 8 + ff as u8) != 0 {
                f[38] = 1.0;
            }
        }

        // Freibauer: auf der Bahn vor ihm steht kein gegnerischer Bauer mehr.
        // „Vor ihm" ist je Farbe eine andere Richtung — dieselbe Quelle von
        // Fehlern wie überall bei vier Spielern, deshalb über
        // `promotion_distance` statt über feste Achsen.
        let rest = promotion_distance(mover, mv.to);
        let mut frei = true;
        for c in Color::ALL {
            if c == mover || !board.active[c.idx()] { continue; }
            let mut bb = board.pieces(c, PieceKind::Pawn);
            while bb != 0 {
                let sq = bb.trailing_zeros() as u8;
                bb &= bb - 1;
                // Gegnerischer Bauer auf derselben Bahn und noch vor uns.
                let gleiche_bahn = match mover {
                    Color::Red | Color::Yellow => file_of(sq) == file_of(mv.to),
                    Color::Blue | Color::Green => rank_of(sq) == rank_of(mv.to),
                };
                if gleiche_bahn && promotion_distance(mover, sq) < rest { frei = false; }
            }
        }
        if frei { f[39] = (7 - rest) as f32 / 7.0; }
    }

    // Wie viele Züge zur Wahl stehen. Kein Merkmal des Zuges, sondern des
    // Kontexts — im Softmax über eine Stellung kürzt es sich weg. Es steht
    // hier, weil das Netz es mit anderen Merkmalen verrechnen kann: ein
    // Schlagzug unter fünf Möglichkeiten wiegt anders als unter vierzig.
    f[25] = (ctx.mover_pieces.len() as f32 / 12.0).min(1.0);

    // ─── Zentrum ─────────────────────────────────────────────────────────────
    let (r, c) = (rank_of(mv.to), file_of(mv.to));
    if (2..=5).contains(&r) && (2..=5).contains(&c) {
        f[11] = if (3..=4).contains(&r) && (3..=4).contains(&c) { 1.0 } else { 0.5 };
    }
}

/// Wie viele Gegner den König des Ziehenden in `after` angreifen.
fn king_attackers_after(after: &Board, mover: Color) -> u32 {
    let king = after.pieces(mover, PieceKind::King);
    if king == 0 { return 0; }   // König geschlagen — im Self-Play möglich
    Color::ALL.iter()
        .filter(|&&c| c != mover && after.active[c.idx()])
        .filter(|&&c| Rules::attacked_squares(after, c) & king != 0)
        .count() as u32
}

/// Vollständiger Merkmalsvektor — für das Training.
///
/// `after` ist die Stellung nach dem Zug; der Aufrufer übergibt sie, weil sie
/// dort meist schon vorliegt.
pub fn features(board: &Board, mv: &Move, after: &Board, ctx: &MoveFeatureContext) -> [f32; N_FEATURES] {
    let mut f = [0.0f32; N_FEATURES];
    let mover = board.to_move;
    features_cheap(board, mv, ctx, &mut f);

    let attackers_after = king_attackers_after(after, mover);
    if ctx.own_king_attackers > 0 && attackers_after == 0 { f[0] = 1.0; }
    if attackers_after < ctx.own_king_attackers {
        f[1] = (ctx.own_king_attackers - attackers_after) as f32;
    }

    // `newly_threatened_kings` zählt Bedrohungen, die es vorher nicht gab —
    // genau die Größe, an der auch die Bonuspunkte hängen (+1 für zwei neue,
    // +5 für drei).
    match Rules::newly_threatened_kings(board, after, mover) {
        0 => {}
        1 => f[7] = 1.0,
        n => { f[7] = 1.0; f[8] = (n - 1) as f32; }
    }
    f
}

/// Merkmale, die sich die Suche leisten kann.
///
/// Zwei Unterschiede zu [`features`]:
///
/// * **`schach_gegeben` und `doppelschach` bleiben null.** Sie brauchen für
///   *jeden* Zug eine frische Angriffskarte des Ziehenden; bei 15 Zügen sind
///   das 15 zusätzliche Zuggenerierungen je Knoten, mehr als eine
///   NNUE-Bewertung kostet. Ihre gelernten Gewichte sind mit +0,09 und +0,66
///   die kleinsten der tatsächlich wirksamen Merkmale — der Handel geht auf.
/// * **Die beiden Schach-Abwehr-Merkmale werden faul berechnet.** Sie sind nur
///   dann von null verschieden, wenn der eigene König überhaupt angegriffen
///   ist, und das steht schon im Kontext. Im Normalfall kostet das nichts; nur
///   im Schach zahlt die Suche den vollen Preis, und dort lohnt es sich.
///
/// Wie viel diese Abkürzung an Vorhersagekraft kostet, misst der Trainer mit
/// (`move_model --help`); gemessen sind es 0,6 Punkte Beam-Trefferquote.
pub fn fast_features(board: &Board, mv: &Move, ctx: &MoveFeatureContext) -> [f32; N_FEATURES] {
    let mut f = [0.0f32; N_FEATURES];
    features_cheap(board, mv, ctx, &mut f);

    if ctx.own_king_attackers > 0 {
        let after = Rules::apply_with_effects(board, *mv);
        let attackers_after = king_attackers_after(&after, board.to_move);
        if attackers_after == 0 { f[0] = 1.0; }
        if attackers_after < ctx.own_king_attackers {
            f[1] = (ctx.own_king_attackers - attackers_after) as f32;
        }
    }
    f
}

// ─── Modell ───────────────────────────────────────────────────────────────────

/// Gelernte Gewichte je Merkmal.
///
/// Der Score eines Zuges ist das Skalarprodukt `w · f`. Für die Sortierung
/// genügt das; die Softmax-Normierung aus dem Training ändert die Reihenfolge
/// nicht, weil der Nenner für alle Züge einer Stellung derselbe ist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveModel {
    pub w: Vec<f32>,
    /// Womit trainiert wurde — steht in der Datei, damit ein Modell später
    /// zuzuordnen ist.
    #[serde(default)]
    pub note: String,
}

/// Die aus echten Partien geschätzten Gewichte, wie die Suche sie benutzt.
///
/// Aus 862.206 Zugentscheidungen von Spielern ab 2400 (6.366 Partien), per
/// konditionaler Logit-Regression; gebaut mit
/// `cargo run --release -p chaturaji-trainer --bin move_model`.
///
/// Es ist die **ungewichtete** Schätzung ohne die beiden teuren Merkmale — also
/// genau das, was [`fast_features`] rechnet. Die erfolgsgewichtete Fassung sagt
/// fast dasselbe (größte Abweichung: Umwandlung +0,18) und sagt den
/// menschlichen Zug einen halben Punkt schlechter vorher.
///
/// Auf zurückgehaltenen Partien gemessen, Anteil der Fälle, in denen der
/// menschliche Zug den Beam der besten 6 überlebt:
///
/// | | Top-1 | im Beam-6 |
/// |---|---|---|
/// | Zufall | 8,0 % | 47,5 % |
/// | die alte Handheuristik | 17,0 % | 54,6 % |
/// | die Gewichte vor dem 2026-09-15 | 25,6 % | 74,3 % |
/// | **diese Gewichte** | **27,2 %** | **75,5 %** |
///
/// Der Sprung kommt aus der Deckung. Vorher konnte das Modell gar nicht sehen,
/// dass eine Figur von der **eigenen** Seite gedeckt wird — `attacked_squares`
/// faltet über generierte Züge, und die gehen nie auf ein eigenes Feld. Damit
/// galt jeder Schlag auf eine gedeckte Figur als frei, und „eingestellt" hieß
/// nur „landet im Beschuss", ohne Rücksicht auf die eigene Deckung. Mit
/// `MoveGen::coverage` stimmt beides, und zwei neue Merkmale kommen hinzu.
///
/// `laesst_haengen` ist mit −3,64 das **gewichtigste Merkmal des Modells** —
/// stärker als jeder Schlagwert. Die Anregung kam aus der Praxis: „Es sollte
/// mehr Gewicht darauf gelegt werden, dass Figuren, die im nächsten Zug
/// angegriffen werden können, gedeckt sind."
///
/// Eingebaut statt als Datei geladen, damit die Engine ohne Fremdpfad
/// auskommt — das Eröffnungsbuch ist optional, die Zugsortierung nicht.
pub const DEFAULT_WEIGHTS: [f32; N_FEATURES] = [
     2.1158,  // koenig_aus_schach
     1.9710,  // schach_vermeiden
     1.1121,  // umwandlung
     0.3909,  // umwandlung_naeher
     3.8331,  // schlag_ungedeckt
     2.5690,  // schlag_gedeckt
     2.2588,  // koenig_geschlagen
     0.0000,  // schach_gegeben   — von `fast_features` nicht berechnet
     0.0000,  // doppelschach     — dito
     1.4039,  // figur_gerettet
    -2.8495,  // figur_eingestellt
    -0.0753,  // zentrum
     0.7208,  // ist_bauer
    -0.2194,  // ist_koenig
     1.3469,  // deckt_bedrohte
    -3.6410,  // laesst_haengen
     0.0000,  // ist_springer     — die Linearform nutzt die Ergänzungen nicht;
     0.0000,  // ist_laeufer         sie sind für das Policy-Netz da
     0.0000,  // ist_boot
     0.0000,  // weg_nach_vorn
     0.0000,  // weg_zur_seite
     0.0000,  // weite
     0.0000,  // zielfeld_gedeckt
     0.0000,  // angreifer_zahl
     0.0000,  // schlaegt_gedeckten
     0.0000,  // zug_zahl
     0.0000,  // greift_an
     0.0000,  // greift_ungedeckt_an
     0.0000,  // beweglichkeit
     0.0000,  // beweglichkeit_gewinn
     0.0000,  // opfer_rang
     0.0000,  // eigener_rang
     0.0000,  // punkte_abstand
     0.0000,  // gegner_zahl
     0.0000,  // see_tausch
     0.0000,  // spiess
     0.0000,  // koenig_naehe
     0.0000,  // feind_koenig_naehe
     0.0000,  // bauer_gedeckt
     0.0000,  // freibauer
];

impl crate::policy::MovePrior for MoveModel {
    fn logit(&self, f: &[f32]) -> f32 {
        self.w.iter().zip(f).map(|(w, x)| w * x).sum()
    }
}

impl Default for MoveModel {
    fn default() -> Self {
        Self { w: DEFAULT_WEIGHTS.to_vec(), note: "eingebaut, aus echten Partien".into() }
    }
}

impl MoveModel {
    pub fn new(w: Vec<f32>, note: impl Into<String>) -> Self {
        Self { w, note: note.into() }
    }

    /// Score aller legalen Züge einer Stellung, absteigend sortiert.
    ///
    /// Der Kontext wird einmal je Stellung gebaut — das ist der teure Teil und
    /// der Grund, warum die Sortierung nicht Zug für Zug nachrechnen darf.
    pub fn rank_moves(&self, board: &Board, moves: &mut [Move]) {
        if moves.len() < 2 { return; }
        let ctx = MoveFeatureContext::new(board);
        let mut mit_score: Vec<(Move, f32)> = moves.iter()
            .map(|mv| (*mv, self.score_features(&fast_features(board, mv, &ctx))))
            .collect();
        mit_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (slot, (mv, _)) in moves.iter_mut().zip(mit_score) { *slot = mv; }
    }

    pub fn zeros() -> Self {
        Self { w: vec![0.0; N_FEATURES], note: String::new() }
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let m: MoveModel = serde_json::from_str(&text)?;
        if m.w.len() != N_FEATURES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Modell hat {} Gewichte, erwartet {N_FEATURES}", m.w.len()),
            ));
        }
        Ok(m)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self)?)
    }

    #[inline]
    pub fn score_features(&self, f: &[f32; N_FEATURES]) -> f32 {
        self.w.iter().zip(f).map(|(w, x)| w * x).sum()
    }

    /// Bequemer Weg für einen einzelnen Zug — berechnet Kontext und Folgestellung
    /// selbst und ist deshalb nur für Einzelabfragen gedacht, nicht in Schleifen.
    pub fn score(&self, board: &Board, mv: &Move) -> f32 {
        let ctx   = MoveFeatureContext::new(board);
        let after = Rules::apply_with_effects(board, *mv);
        self.score_features(&features(board, mv, &after, &ctx))
    }

    /// Gewichtstabelle als Text, absteigend nach Betrag.
    pub fn table(&self) -> String {
        let mut idx: Vec<usize> = (0..N_FEATURES).collect();
        idx.sort_by(|&a, &b| self.w[b].abs().partial_cmp(&self.w[a].abs()).unwrap());
        let mut s = String::new();
        for i in idx {
            s.push_str(&format!("  {:>18}  {:+8.4}\n", FEATURE_NAMES[i], self.w[i]));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chaturaji_core::board::sq;

    /// Die Umwandlungsrichtung ist je Farbe eine andere — Rot Nord, Blau Ost,
    /// Gelb Süd, Grün West. Bei vier Spielern die klassische Fehlerquelle.
    #[test]
    fn promotion_distance_points_the_right_way_for_all_four() {
        // a1 = 0, h1 = 7, a8 = 56, h8 = 63.
        assert_eq!(promotion_distance(Color::Red,    sq(0, 0)), 7, "Rot zieht nach Norden");
        assert_eq!(promotion_distance(Color::Red,    sq(0, 7)), 0);
        assert_eq!(promotion_distance(Color::Blue,   sq(0, 0)), 7, "Blau zieht nach Osten");
        assert_eq!(promotion_distance(Color::Blue,   sq(7, 0)), 0);
        assert_eq!(promotion_distance(Color::Yellow, sq(0, 7)), 7, "Gelb zieht nach Süden");
        assert_eq!(promotion_distance(Color::Yellow, sq(0, 0)), 0);
        assert_eq!(promotion_distance(Color::Green,  sq(7, 0)), 7, "Grün zieht nach Westen");
        assert_eq!(promotion_distance(Color::Green,  sq(0, 0)), 0);
    }

    /// In der Startstellung darf kein einziger Zug ein Schach-, Schlag- oder
    /// Umwandlungsmerkmal setzen. Wenn doch, stimmt etwas Grundsätzliches nicht.
    #[test]
    fn opening_moves_set_no_tactical_features() {
        let board = Board::default();
        let ctx   = MoveFeatureContext::new(&board);
        for mv in Rules::legal_moves(&board) {
            let after = Rules::apply_with_effects(&board, mv);
            let f = features(&board, &mv, &after, &ctx);
            for i in [0usize, 2, 4, 5, 6, 8] {
                assert_eq!(f[i], 0.0, "{} bei {:?} in der Startstellung", FEATURE_NAMES[i], mv);
            }
        }
    }

    /// Ein Bauernzug nach vorn muss „näher an die Umwandlung" setzen, und der
    /// Wert muss mit der Nähe wachsen.
    #[test]
    fn advancing_a_pawn_scores_progress_that_grows_near_the_line() {
        let board = Board::default();
        let ctx   = MoveFeatureContext::new(&board);
        // Rot am Zug: irgendein Bauernzug nach Norden.
        let mv = Rules::legal_moves(&board).into_iter()
            .find(|m| board.piece_at(m.from).map(|p| p.kind) == Some(PieceKind::Pawn)
                      && rank_of(m.to) > rank_of(m.from))
            .expect("Rot hat Bauernzüge nach vorn");
        let after = Rules::apply_with_effects(&board, mv);
        let f = features(&board, &mv, &after, &ctx);
        assert!(f[3] > 0.0, "umwandlung_naeher muss gesetzt sein");
        assert_eq!(f[12], 1.0, "ist_bauer muss gesetzt sein");

        // Näher an der Linie ⇒ größerer Wert.
        let nah  = (7 - promotion_distance(Color::Red, sq(0, 6))) as f32 / 7.0;
        let fern = (7 - promotion_distance(Color::Red, sq(0, 2))) as f32 / 7.0;
        assert!(nah > fern, "je näher an der Umwandlung, desto höher");
    }

    /// Schlagen muss in genau einem der beiden Schlagfelder landen — gedeckt
    /// oder ungedeckt, nie in beiden und nie in keinem.
    #[test]
    fn a_capture_lands_in_exactly_one_of_the_two_capture_slots() {
        let mut board = Board::default();
        let mut gesehen_gedeckt = false;
        let mut gesehen_frei    = false;

        // Ein paar Halbzüge spielen, bis Schläge auftauchen.
        for _ in 0..60 {
            let ctx   = MoveFeatureContext::new(&board);
            let moves = Rules::legal_moves(&board);
            if moves.is_empty() { break; }
            for mv in &moves {
                if mv.captured.is_none() { continue; }
                let after = Rules::apply_with_effects(&board, *mv);
                let f = features(&board, mv, &after, &ctx);
                assert!(
                    (f[4] > 0.0) ^ (f[5] > 0.0),
                    "genau eines von schlag_ungedeckt/schlag_gedeckt: {:?} / {:?}", f[4], f[5],
                );
                if f[5] > 0.0 { gesehen_gedeckt = true; } else { gesehen_frei = true; }
            }
            board = Rules::apply_with_effects(&board, moves[moves.len() / 2]);
        }
        assert!(gesehen_gedeckt || gesehen_frei, "in 60 Halbzügen kam kein Schlag vor");
    }

    /// Das Modell darf die Reihenfolge nur über die Gewichte bestimmen: mit
    /// Nullgewichten sind alle Züge gleichwertig.
    #[test]
    fn zero_weights_score_everything_equally() {
        let board = Board::default();
        let m = MoveModel::zeros();
        for mv in Rules::legal_moves(&board) {
            assert_eq!(m.score(&board, &mv), 0.0);
        }
    }

    #[test]
    fn a_model_survives_a_round_trip_through_json() {
        let dir = std::env::temp_dir().join("chaturaji-move-model-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w.json");
        let p = path.to_str().unwrap();

        let m = MoveModel::new((0..N_FEATURES).map(|i| i as f32 * 0.1).collect(), "test");
        m.save(p).unwrap();
        let gelesen = MoveModel::load(p).unwrap();
        assert_eq!(gelesen.w, m.w);
        assert_eq!(gelesen.note, "test");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn a_model_with_the_wrong_width_is_rejected() {
        let dir = std::env::temp_dir().join("chaturaji-move-model-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kaputt.json");
        let p = path.to_str().unwrap();
        std::fs::write(p, r#"{"w":[1.0,2.0],"note":""}"#).unwrap();
        assert!(MoveModel::load(p).is_err(), "zu kurzer Gewichtsvektor muss abgelehnt werden");
        std::fs::remove_file(p).ok();
    }
}
