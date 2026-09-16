<?php
/**
 * Speichert eine Aufzeichnung (Recording) aus der Chaturaji-Web-UI.
 *
 * Erwartet POST mit JSON-Body: { "filename": "<name>.json", "game": { … } }
 *   - Schreibt das Spiel-JSON nach  game_analysis/<filename>
 *   - Ergänzt/ersetzt einen Eintrag in  games_index.json
 * damit das Spiel in der App-Liste erscheint und wie jedes andere ladbar ist.
 *
 * Pendant zur lokalen Node-Route POST /save-game.php in server.js.
 */

header('Content-Type: application/json; charset=utf-8');

// Aufrufe vom Bookmarklet kommen aus einer chess.com-Seite, sind also
// site-fremd. Ohne diese Köpfe schickt der Browser die Anfrage zwar ab, hält
// die Antwort aber zurück, und das Bookmarklet könnte nicht sagen, ob das
// Speichern geklappt hat.
//
// **Diese Köpfe schützen nichts.** `Access-Control-Allow-Origin` weist keine
// Anfrage ab; es sagt dem *Browser* nur, ob er die Antwort an das aufrufende
// Skript weiterreichen darf. Die Anfrage selbst kommt an und wird ausgeführt,
// und wer nicht im Browser sitzt — curl, ein Skript, irgendein Server —
// ignoriert CORS vollständig. Bis 2026-09-16 konnte deshalb jeder beliebige
// Dateien hier ablegen; nachgewiesen mit einem curl-Aufruf ohne Herkunft. Was
// den Zugang regelt, ist das Token weiter unten.
$erlaubt = ['https://www.chess.com', 'https://chess.com'];
$herkunft = $_SERVER['HTTP_ORIGIN'] ?? '';
if (in_array($herkunft, $erlaubt, true)) {
    header('Access-Control-Allow-Origin: ' . $herkunft);
    header('Vary: Origin');
    header('Access-Control-Allow-Headers: Content-Type, X-Chaturaji-Token');
    header('Access-Control-Allow-Methods: POST, OPTIONS');
    header('Access-Control-Max-Age: 86400');
}

// Der Vorab-Check des Browsers (wegen Content-Type: application/json) muss vor
// der Methodenprüfung beantwortet werden, sonst bekäme er ein 405 und der
// eigentliche POST fände nie statt.
if (($_SERVER['REQUEST_METHOD'] ?? '') === 'OPTIONS') {
    http_response_code(204);
    exit;
}

function fail(int $code, string $msg): void {
    http_response_code($code);
    echo json_encode(['ok' => false, 'error' => $msg], JSON_UNESCAPED_UNICODE);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    fail(405, 'Nur POST erlaubt');
}

// ─── Zugang ──────────────────────────────────────────────────────────────────
//
// Das Token liegt in `token.php`, einer Datei, die es zurückgibt statt es
// auszugeben: wer sie im Browser aufruft, bekommt eine leere Seite, weil PHP
// sie ausführt. Sie steht in `.gitignore` und wird nur per FTP hochgeladen.
//
// Fehlt sie, wird **abgewiesen**, nicht durchgelassen. Ein vergessenes
// Hochladen soll das Speichern hörbar kaputtmachen und nicht still die Tür
// wieder öffnen.
// Der Körper wird einmal gelesen und weitergereicht. `php://input` mehrfach
// zu öffnen ist je nach PHP-Aufbau nicht verlässlich.
$raw = file_get_contents('php://input');
if ($raw === false || $raw === '') {
    fail(400, 'Leerer Request-Body');
}

// Obergrenze. Die größte der 6.367 vorhandenen Analysen misst 157 KB, im
// Mittel sind es 44 KB; ein Megabyte ist reichlich und begrenzt trotzdem, was
// ein einzelner Aufruf anrichten kann.
const MAX_BYTES = 1048576;
if (strlen($raw) > MAX_BYTES) {
    fail(413, 'Spiel-Objekt zu groß (über ' . (MAX_BYTES / 1024) . ' KB).');
}

$tokenDatei = __DIR__ . '/token.php';
$erwartet = is_file($tokenDatei) ? (include $tokenDatei) : null;
if (!is_string($erwartet) || strlen($erwartet) < 16) {
    fail(503, 'Kein Token hinterlegt — token.php fehlt oder ist unbrauchbar.');
}

// Kopf bevorzugt; das Feld im Körper ist für Aufrufer ohne eigene Kopfzeilen.
$gesendet = $_SERVER['HTTP_X_CHATURAJI_TOKEN'] ?? '';

// `hash_equals` statt `===`: der Vergleich läuft in konstanter Zeit und
// verrät über die Dauer nicht, wie viele Zeichen schon stimmten.
if (!is_string($gesendet) || !hash_equals($erwartet, $gesendet)) {
    // Der Körper darf ein `token`-Feld tragen, für Aufrufer ohne eigene
    // Kopfzeilen.
    $vorab = json_decode($raw, true);
    $imKoerper = is_array($vorab) ? ($vorab['token'] ?? '') : '';
    if (!is_string($imKoerper) || !hash_equals($erwartet, $imKoerper)) {
        fail(403, 'Token fehlt oder stimmt nicht.');
    }
}

$payload = json_decode($raw, true);
if (!is_array($payload)) {
    fail(400, 'Ungültiges JSON');
}

$filename = $payload['filename'] ?? '';
$game     = $payload['game'] ?? null;
$baseFile = $payload['baseFile'] ?? null; // gesetzt, wenn ein bestehendes Spiel aufgezeichnet wurde

// Dateiname streng validieren. Erlaubt ist nur, was die App selbst erzeugt:
//
//   108644945.json        eine heruntergeladene Partie
//   108644945-1.json      eine Aufzeichnung als Variante davon
//   recording-1758…json   eine frisch aufgezeichnete Partie
//
// Vorher galt `[\w.-]+\.json`, also praktisch jeder Name — damit ließe sich
// eine bestehende Analyse unter ihrem eigenen Namen überschreiben. Alle 6.367
// vorhandenen Dateien passen in das engere Muster.
if (!is_string($filename)
    || strpos($filename, '..') !== false
    || !preg_match('/^(\d+(-\d+)?|recording-\d+)\.json$/', $filename)) {
    fail(400, 'Ungültiger Dateiname');
}
$filename = basename($filename); // doppelt absichern
if (!is_array($game)) {
    fail(400, 'Ungültiges Spiel-Objekt');
}

$analysisDir = __DIR__ . '/game_analysis';
$indexFile   = __DIR__ . '/games_index.json';

if (!is_dir($analysisDir) && !@mkdir($analysisDir, 0775, true)) {
    fail(500, 'game_analysis/ nicht anlegbar');
}

// games_index.json laden (oder leer starten)
$index = [];
if (is_file($indexFile)) {
    $decoded = json_decode(@file_get_contents($indexFile), true);
    if (is_array($decoded)) {
        $index = $decoded;
    }
}

// Aufzeichnung eines bestehenden Spiels: Dateiname = alter Name (ohne evtl.
// bestehende "-<Nr>"-Endung) plus fortlaufende Nummer. games_index.json wird
// gelesen, um bereits gespeicherte Varianten zu überspringen.
if (is_string($baseFile)
    && strpos($baseFile, '..') === false
    && preg_match('/^[\w.-]+\.json$/', $baseFile)) {
    $root = preg_replace('/-\d+$/', '', preg_replace('/\.json$/i', '', basename($baseFile)));
    $re   = '/^' . preg_quote($root, '/') . '-(\d+)\.json$/i';
    $max  = 0;
    foreach ($index as $e) {
        if (isset($e['file']) && is_string($e['file']) && preg_match($re, $e['file'], $m)) {
            $max = max($max, (int) $m[1]);
        }
    }
    $filename = $root . '-' . ($max + 1) . '.json';
}

// Spiel-JSON schreiben
$gameJson = json_encode(
    $game,
    JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES
);
if (@file_put_contents($analysisDir . '/' . $filename, $gameJson) === false) {
    fail(500, 'Schreiben von ' . $filename . ' fehlgeschlagen (Rechte?)');
}

// Index-Eintrag im selben Format wie build_games_index.py
$entry = [
    'file'       => $filename,
    'gameNr'     => $game['gameNr'] ?? null,
    'date'       => gmdate('Y-m-d\TH:i:s.000\Z'),
    'result'     => $game['result'] ?? '',
    'players'    => [
        $game['username1'] ?? '', $game['username2'] ?? '',
        $game['username3'] ?? '', $game['username4'] ?? '',
    ],
    'ratings'    => [
        $game['rating1'] ?? null, $game['rating2'] ?? null,
        $game['rating3'] ?? null, $game['rating4'] ?? null,
    ],
    'placements' => [null, null, null, null],
];

// Bestehenden Eintrag mit gleichem Dateinamen ersetzen, sonst anhängen
$replaced = false;
foreach ($index as $i => $e) {
    if (isset($e['file']) && $e['file'] === $filename) {
        $index[$i] = $entry;
        $replaced = true;
        break;
    }
}
if (!$replaced) {
    $index[] = $entry;
}

if (@file_put_contents(
        $indexFile,
        json_encode($index, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES)
    ) === false) {
    fail(500, 'games_index.json nicht schreibbar (Rechte?)');
}

echo json_encode(['ok' => true, 'file' => $filename], JSON_UNESCAPED_UNICODE);
