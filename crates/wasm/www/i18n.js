// Mehrsprachigkeit: lädt JSON-Wörterbücher unter i18n/<lang>.json und ersetzt
// Marker im DOM. Marker:
//   data-i18n="key"               → textContent
//   data-i18n-html="key"          → innerHTML (für Strings mit HTML)
//   data-i18n-title="key"         → title-Attribut
//   data-i18n-placeholder="key"   → placeholder-Attribut
//   data-i18n-aria-label="key"    → aria-label-Attribut
//
// Dynamische Strings im JS holt man sich per t('key', { var: 'wert' }).
// Auf Sprachwechsel kann man via onLanguageChange(fn) reagieren.

// Flagge je Sprache (Unicode-Emoji aus Regional-Indikator-Buchstaben).
// Hinweis: Windows zeigt mangels Glyphen den Ländercode ("GB", "DE", …) statt
// einer Flagge — alle anderen großen Plattformen rendern korrekt.
export const SUPPORTED_LANGS = [
  { code: 'en', label: 'English',  flag: '🇬🇧' },
  { code: 'de', label: 'Deutsch',  flag: '🇩🇪' },
  { code: 'fr', label: 'Français', flag: '🇫🇷' },
  { code: 'es', label: 'Español',  flag: '🇪🇸' },
  { code: 'he', label: 'עברית',   flag: '🇮🇱' },
  { code: 'ar', label: 'العربية', flag: '🇸🇦' },
];

const RTL_LANGS = new Set(['he', 'ar']);

const dictionaries = {};
let currentLang = 'en';
const listeners = [];

async function loadLang(lang) {
  if (dictionaries[lang]) return dictionaries[lang];
  const r = await fetch('i18n/' + lang + '.json?v=2605141');
  if (!r.ok) throw new Error('language file not found: ' + lang);
  const data = await r.json();
  dictionaries[lang] = data;
  return data;
}

function format(template, params) {
  if (params == null || typeof template !== 'string') return template;
  return template.replace(/\{(\w+)\}/g, (_, k) => (k in params ? params[k] : '{' + k + '}'));
}

export function t(key, params) {
  const dict = dictionaries[currentLang] || {};
  const fallback = dictionaries['en'] || {};
  const value = (key in dict) ? dict[key] : (key in fallback ? fallback[key] : key);
  return format(value, params);
}

export function getLang() { return currentLang; }

export function onLanguageChange(fn) { listeners.push(fn); }

export function applyTranslations(root = document) {
  for (const el of root.querySelectorAll('[data-i18n]')) {
    el.textContent = t(el.getAttribute('data-i18n'));
  }
  for (const el of root.querySelectorAll('[data-i18n-html]')) {
    el.innerHTML = t(el.getAttribute('data-i18n-html'));
  }
  for (const el of root.querySelectorAll('[data-i18n-title]')) {
    el.title = t(el.getAttribute('data-i18n-title'));
  }
  for (const el of root.querySelectorAll('[data-i18n-placeholder]')) {
    el.placeholder = t(el.getAttribute('data-i18n-placeholder'));
  }
  for (const el of root.querySelectorAll('[data-i18n-aria-label]')) {
    el.setAttribute('aria-label', t(el.getAttribute('data-i18n-aria-label')));
  }
  document.documentElement.lang = currentLang;
  //document.documentElement.dir  = RTL_LANGS.has(currentLang) ? 'rtl' : 'ltr';
  // <title> der Seite mitziehen, falls Key vorhanden
  const titleKey = document.documentElement.getAttribute('data-i18n-title-tag');
  if (titleKey) document.title = t(titleKey);
}

export async function setLang(lang) {
  if (!SUPPORTED_LANGS.some(l => l.code === lang)) lang = 'en';
  await loadLang(lang);
  currentLang = lang;
  try { localStorage.setItem('uiLang', lang); } catch (_) {}
  applyTranslations();
  for (const fn of listeners) {
    try { fn(lang); } catch (e) { console.error('languageChange listener:', e); }
  }
}

export async function initI18n(defaultLang = 'en') {
  let lang = defaultLang;
  try {
    const stored = localStorage.getItem('uiLang');
    if (stored) lang = stored;
  } catch (_) {}
  if (!SUPPORTED_LANGS.some(l => l.code === lang)) lang = defaultLang;
  // Englisch als Fallback immer mitladen, damit fehlende Keys aufgefangen werden.
  if (lang !== 'en') { try { await loadLang('en'); } catch (_) {} }
  await loadLang(lang);
  currentLang = lang;
  applyTranslations();
}
