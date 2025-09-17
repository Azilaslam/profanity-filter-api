# main.py
import os
import re
import random
import pathlib
import itertools
import unicodedata
import asyncio
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache, partial

app = FastAPI(title="Profanity Filter API (multilingual + stretch handling + normalizer)")

# Backend secret for RapidAPI/Render
BACKEND_SECRET: Optional[str] = os.getenv("BACKEND_SECRET")

# ---------------------------
# Try to import confusable_homoglyphs (best effort)
# ---------------------------
USE_CONFUSABLE_LIB = False
try:
    import confusable_homoglyphs.confusables as confusables_lib
    USE_CONFUSABLE_LIB = True
except Exception:
    confusables_lib = None
    USE_CONFUSABLE_LIB = False

# ---------------------------
# Middleware: simple backend secret guard
# ---------------------------
@app.middleware("http")
async def verify_backend_secret(request: Request, call_next):
    path = request.url.path or ""
    if request.method == "OPTIONS" or path.startswith("/docs") or path.startswith("/openapi.json") or path.startswith("/redoc"):
        return await call_next(request)
    if BACKEND_SECRET:
        header_secret = request.headers.get("x-backend-secret")
        if header_secret != BACKEND_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    return await call_next(request)

# ---------------------------
# Request models
# ---------------------------
class CensorIn(BaseModel):
    text: str
    style: Optional[str] = "stars"    # "stars" | "symbols" | "mask" | "blocks" | "red"
    mode: Optional[str] = "all"      # "all" or "extreme"
    normalize_homoglyphs: Optional[bool] = False
    custom_blocklist: Optional[List[str]] = None
    metadata: Optional[bool] = True   # whether to return rich metadata

class BatchIn(BaseModel):
    items: List[CensorIn]

# ---------------------------
# Health
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------------------
# Load word files
# ---------------------------
def load_wordfile(name: str, comma_separated: bool = False):
    p = pathlib.Path(__file__).parent / name
    words = set()
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
            if comma_separated:
                raw = [w.strip().lower() for w in content.split(",")]
            else:
                raw = [line.strip().lower() for line in content.splitlines()]
            words = {w for w in raw if w}
    return words

BAD_ALL = load_wordfile("badwords_all.txt", comma_separated=False)
BAD_EXTREME = load_wordfile("badwords_extreme.txt", comma_separated=True)
BAD_MISC = load_wordfile("badwords_misc.txt", comma_separated=False)
BAD_ALL |= BAD_MISC
BAD_EXTREME |= BAD_MISC

if not BAD_ALL:
    BAD_ALL = {"shit","fuck","bitch","idiot","loser","simp","gooner"}
if not BAD_EXTREME:
    BAD_EXTREME = {"kys","rape","nigger","faggot"}
if not BAD_MISC:
    BAD_MISC = {"puta","merda","merde","cabron","connard"}

# ---------------------------
# Normalization helpers (diacritics, collapsed repeats)
# ---------------------------
def normalize_lookup_word(w: str) -> str:
    nw = unicodedata.normalize("NFKD", w)
    nw = "".join(ch for ch in nw if not unicodedata.combining(ch))
    return nw.casefold()  # use casefold for better Unicode case handling

def collapse_repeats_to_one(s: str) -> str:
    return re.sub(r'(.)\1+', r'\1', s, flags=re.IGNORECASE)

# Precompute lookup sets using casefolded normalized forms
BAD_ALL_LOOKUP = {normalize_lookup_word(w) for w in BAD_ALL}
BAD_EXTREME_LOOKUP = {normalize_lookup_word(w) for w in BAD_EXTREME}
BAD_ALL_COLLAPSED = {collapse_repeats_to_one(normalize_lookup_word(w)) for w in BAD_ALL}
BAD_EXTREME_COLLAPSED = {collapse_repeats_to_one(normalize_lookup_word(w)) for w in BAD_EXTREME}

SYMBOLS = ["@", "#", "$", "!", "%", "&"]

# ---------------------------
# censor styles
# ---------------------------
def censor_word(word: str, style: str) -> str:
    if style == "stars":
        return word[0] + "*"*(len(word)-1) if len(word) > 0 else word
    elif style == "symbols":
        return "".join(random.choice(SYMBOLS) for _ in word)
    elif style == "mask":
        if len(word) <= 2:
            return "*"*len(word)
        return word[0] + "*"*(len(word)-2) + word[-1]
    elif style == "blocks":
        BLOCKS = ["🟥","🟩","🟦","🟨","🟪","⬛","⬜"]
        return "".join(random.choice(BLOCKS) for _ in word)
    elif style == "red":
        return "🟥" * len(word)
    return "*"*len(word)

# ---------------------------
# LEET_MAP and fuzzy pattern builder
# ---------------------------
LEET_MAP = {
    "a": ["a", "@", "4", "à", "á", "â", "ä", "ã", "å", "ā"],
    "b": ["b", "8"],
    "c": ["c", "(", "<", "ç"],
    "d": ["d"],
    "e": ["e", "3", "è", "é", "ê", "ë", "ē"],
    "f": ["f"],
    "g": ["g", "9"],
    "h": ["h", "#"],
    "i": ["i", "1", "!", "|", "ì", "í", "î", "ï", "ī"],
    "j": ["j"],
    "k": ["k"],
    "l": ["l", "1", "|"],
    "m": ["m"],
    "n": ["n", "ñ"],
    "o": ["o", "0", "ò", "ó", "ô", "ö", "õ", "ō"],
    "p": ["p"],
    "q": ["q"],
    "r": ["r"],
    "s": ["s", "$", "5", "š"],
    "t": ["t", "7", "+", "ţ"],
    "u": ["u", "v", "@", "ù", "ú", "û", "ü", "ū"],
    "v": ["v", "\\/"],
    "w": ["w"],
    "x": ["x", "%"],
    "y": ["y", "ý", "ÿ"],
    "z": ["z", "2", "ž"],
}

def char_class_for(ch: str) -> str:
    ch_low = ch.lower()
    if ch_low in LEET_MAP:
        parts = [re.escape(x) for x in set(LEET_MAP[ch_low])]
        parts = sorted(parts, key=lambda s: -len(s))
        return "(" + "|".join(parts) + ")"
    else:
        return re.escape(ch)

def build_fuzzy_pattern(word: str) -> str:
    tokens = word.split()
    token_patterns = []
    for token in tokens:
        parts = []
        for i, ch in enumerate(token):
            if ch.lower() in "aeiou":
                parts.append(char_class_for(ch) + "{1,2}")
            else:
                if i == len(token)-1:
                    parts.append(char_class_for(ch) + "{1,5}")
                else:
                    parts.append(char_class_for(ch))
            parts.append(r"[\W_]{0,3}")
        token_pattern = "".join(parts[:-1])
        token_patterns.append(token_pattern)
    inner = r"[\W_]*".join(token_patterns)
    return r"(?<!\w)" + inner + r"(?!\w)"

def generate_variants(word: str):
    variants = {word}
    vowels = "aeiou"
    if word and word[0].lower() in vowels:
        variants.add(word[1:])
    return variants

# ---------------------------
# Precompile patterns (with caution)
# ---------------------------
PRECOMPILED = {"all": [], "extreme": []}

def compile_patterns():
    for mode, bad_set in [("all", BAD_ALL), ("extreme", BAD_EXTREME)]:
        patterns = []
        for w in sorted(bad_set, key=len, reverse=True):
            for variant in generate_variants(w):
                try:
                    regex = re.compile(build_fuzzy_pattern(variant), flags=re.IGNORECASE)
                    patterns.append(regex)
                except re.error:
                    try:
                        regex = re.compile(re.escape(variant), flags=re.IGNORECASE)
                        patterns.append(regex)
                    except Exception:
                        continue
        PRECOMPILED[mode] = patterns

compile_patterns()

# ---------------------------
# Remove middle-finger emoji helper
# ---------------------------
def remove_middle_fingers(text: str) -> str:
    out_chars = []
    skip_skin_tone = False
    for ch in text:
        if ch == "\U0001F595":  # 🖕
            skip_skin_tone = True
            continue
        if skip_skin_tone and "\U0001F3FB" <= ch <= "\U0001F3FF":
            skip_skin_tone = False
            continue
        skip_skin_tone = False
        out_chars.append(ch)
    return "".join(out_chars)

# ---------------------------
# Repeated-stretch helpers (kept with sensible caps)
# ---------------------------
MAX_REDUCTION_PER_BLOCK = 5
MAX_VARIANTS_TOTAL = 200
VOWELS = set("aeiouAEIOU")

def is_vowel(ch: str):
    return ch.lower() in VOWELS

def is_consonant(ch: str):
    return ch.isalpha() and ch.lower() not in VOWELS

def find_alpha_runs(word: str):
    runs = []
    i = 0
    while i < len(word):
        if not word[i].isalpha():
            i += 1
            continue
        j = i + 1
        while j < len(word) and word[j].lower() == word[i].lower():
            j += 1
        runs.append((i, j - i, word[i]))
        i = j
    return runs

def generate_reduction_variants(original: str, cap_per_block=MAX_REDUCTION_PER_BLOCK, max_total=MAX_VARIANTS_TOTAL):
    runs = find_alpha_runs(original)
    target_runs = [(pos, length, ch) for (pos, length, ch) in runs if length >= 2]
    if not target_runs:
        return []

    choices_per_run = []
    for pos, length, ch in target_runs:
        if is_vowel(ch):
            opts = []
            if length > 2:
                opts.append(2)
                if length > 3:
                    opts.append(1)
            else:
                opts = [2, 1]
            choices_per_run.append(sorted(set(opts)))
        elif is_consonant(ch):
            opts = list(range(max(1, length - cap_per_block), length + 1))
            choices_per_run.append(sorted(set(opts), reverse=True))
        else:
            choices_per_run.append([length])

    variants = []
    seen = set()
    for chosen in itertools.product(*choices_per_run):
        w = list(original)
        for (pos, orig_len, ch), keep_len in sorted(zip(target_runs, chosen), key=lambda x: x[0][0], reverse=True):
            del w[pos:pos+orig_len]
            w[pos:pos] = [ch] * keep_len
        candidate = "".join(w)
        if candidate not in seen:
            variants.append(candidate)
            seen.add(candidate)
        if len(variants) >= max_total:
            break
    return variants

def variant_matches_lookup(candidate: str, mode: str) -> bool:
    n = normalize_lookup_word(candidate)
    collapsed = collapse_repeats_to_one(n)
    if mode == "extreme":
        if n in BAD_EXTREME_LOOKUP or collapsed in BAD_EXTREME_COLLAPSED:
            return True
        return False
    else:
        if n in BAD_ALL_LOOKUP or n in BAD_EXTREME_LOOKUP:
            return True
        if collapsed in BAD_ALL_COLLAPSED or collapsed in BAD_EXTREME_COLLAPSED:
            return True
        return False

def pre_censor_repeated_stretches(text: str, mode: str, style: str, matched_terms: List[str]) -> str:
    """
    Pre-pass that censors tokens with stretched letters when a reduced variant matches lookup.
    This version records matched terms into `matched_terms` when it decides to censor a token.
    """
    token_re = re.compile(r"\b[\w@#\$%!\|\-']+\b", flags=re.UNICODE)

    def replace_token(m):
        token = m.group(0)
        if len(token) < 3:
            return token
        runs = find_alpha_runs(token)
        target_runs = [(pos, length, ch) for (pos, length, ch) in runs if length >= 2 and ch.isalpha()]
        if not target_runs:
            return token

        choices = []
        for pos, length, ch in target_runs:
            max_keep = length
            min_keep = 1
            opts = list(range(max_keep, min_keep - 1, -1))
            if len(opts) > MAX_REDUCTION_PER_BLOCK:
                opts = opts[:MAX_REDUCTION_PER_BLOCK]
            choices.append(opts)

        total_possible = 1
        for c in choices:
            total_possible *= len(c)
            if total_possible > MAX_VARIANTS_TOTAL:
                break
        if total_possible > MAX_VARIANTS_TOTAL:
            choices = []
            for pos, length, ch in target_runs:
                opts = [length]
                if length > 1:
                    opts.append(1)
                choices.append(opts)

        variants_tried = 0
        seen = set()
        for chosen in itertools.product(*choices):
            w = list(token)
            for (pos, orig_len, ch), keep_len in sorted(zip(target_runs, chosen), key=lambda x: x[0][0], reverse=True):
                del w[pos:pos+orig_len]
                w[pos:pos] = [ch] * keep_len
            candidate = "".join(w)
            if candidate in seen:
                continue
            seen.add(candidate)
            variants_tried += 1
            if variant_matches_lookup(candidate, mode):
                # record the normalized matched term (for observability)
                matched_terms.append(normalize_lookup_word(candidate))
                return censor_word(token, style)
            if variants_tried >= MAX_VARIANTS_TOTAL:
                break

        return token

    return token_re.sub(replace_token, text)

def final_sanitize(text: str, style: str, mode: str, matched_terms: List[str]) -> str:
    patterns = PRECOMPILED.get(mode, [])
    for regex in patterns:
        def repl(m):
            matched_terms.append(m.group(0))
            return censor_word(m.group(0), style)
        text = regex.sub(repl, text)
    return text

# ---------------------------
# Small confusable/homoglyph fallback map (curated)
# We'll use confusable_homoglyphs library when available for better coverage.
# ---------------------------
CONFUSABLES = {
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y", "к": "k",
    "і": "i", "ѕ": "s", "т": "t", "м": "m", "в": "v",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "Ａ":"A","Ｂ":"B","Ｃ":"C", "ᴡ": "w",
    "ᴛ": "t",
    "ᴀ": "a",
    "ʟ": "l",
    "ʀ": "r",
    "ɢ": "g",
}

def normalize_confusables(s: str) -> str:
    """
    Normalize confusable characters to recommended ASCII-like equivalents.
    If confusable_homoglyphs is installed, use it; otherwise fall back to curated map.
    """
    if USE_CONFUSABLE_LIB and confusables_lib:
        try:
            found = confusables_lib.is_confusable(s, greedy=True, preferred_aliases=['latin'])
            if not found:
                return s
            out = list(s)
            for item in found:
                ch = item.get('character')
                homos = item.get('homoglyphs') or []
                if not homos:
                    continue
                mapped = None
                # homos is list; prefer dict with 'c' key
                if isinstance(homos[0], dict):
                    mapped = homos[0].get('c')
                elif isinstance(homos[0], str):
                    mapped = homos[0]
                if mapped:
                    out = [mapped if c == ch else mapped.upper() if c == ch.upper() and mapped.islower() else c for c in out]
            return "".join(out)
        except Exception:
            pass

    # fallback
    out = []
    for ch in s:
        mapped = CONFUSABLES.get(ch)
        if mapped is None:
            mapped = CONFUSABLES.get(ch.lower())
            if mapped and ch.isupper():
                out.append(mapped.upper())
                continue
        if mapped:
            out.append(mapped)
        else:
            out.append(ch)
    return "".join(out)

# ---------------------------
# Core censor (pure CPU function). Returns structured metadata dict.
# We'll wrap with caching and call it in a threadpool in the async endpoint.
# ---------------------------
def smart_censor_unwrapped(original_text: str, style: str, mode: str,
                           normalize_homoglyphs: bool = False,
                           custom_blocklist: Optional[List[str]] = None) -> Dict[str, Any]:

    # Defensive: cap very large inputs (to avoid catastrophic regex). Decide policy:
    MAX_INPUT_LENGTH = 4000
    text = original_text
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]
        truncated = True
    else:
        truncated = False

    reasons: List[str] = []
    matched_terms: List[str] = []   # normalized matched terms (casefolded)
    raw_matched_terms: List[Tuple[str, str]] = []  # (raw, source) for debugging if needed

    # 0) optional confusable normalization
    maybe_normalized = text
    if normalize_homoglyphs:
        maybe_normalized = normalize_confusables(maybe_normalized)
        reasons.append("normalized_homoglyphs")

    # 1) Unicode normalization
    maybe_normalized = unicodedata.normalize("NFKC", maybe_normalized)

    # 2) Remove middle-finger emoji (record reason if removed)
    without_fingers = remove_middle_fingers(maybe_normalized)
    if without_fingers != maybe_normalized:
        reasons.append("emoji_middle_finger")

    # 3) Pre-pass: stretched-letter reduction check (this will append normalized matches)
    pre = pre_censor_repeated_stretches(without_fingers, mode, style, matched_terms)
    if pre != without_fingers:
        reasons.append("stretched_variant")

    # 4) Custom substring / fuzzy blocking (PERMISSIVE + pre-pass stretched reduction)
    out = pre
    custom_matched: List[str] = []
    if custom_blocklist:
        # Prepare normalized custom tokens
        norm_customs = []
        for token in custom_blocklist:
            if not token:
                continue
            tok_norm = token
            if normalize_homoglyphs:
                tok_norm = normalize_confusables(tok_norm)
            tok_norm = unicodedata.normalize("NFKC", tok_norm)
            tok_norm = tok_norm.casefold()
            norm_customs.append((token, tok_norm))

        # 4a) First: pre-pass stretched reduction check for custom tokens (scan tokens like pre-pass does)
        token_re = re.compile(r"\b[\w@#\$%!\|\-']+\b", flags=re.UNICODE)
        def _check_and_censor_token_for_custom(m):
            token = m.group(0)
            if len(token) < 3:
                return token
            runs = find_alpha_runs(token)
            target_runs = [(pos, length, ch) for (pos, length, ch) in runs if length >= 2 and ch.isalpha()]
            if not target_runs:
                return token
            choices = []
            for pos, length, ch in target_runs:
                max_keep = length
                min_keep = 1
                opts = list(range(max_keep, min_keep - 1, -1))
                if len(opts) > MAX_REDUCTION_PER_BLOCK:
                    opts = opts[:MAX_REDUCTION_PER_BLOCK]
                choices.append(opts)
            total_possible = 1
            for c in choices:
                total_possible *= len(c)
                if total_possible > MAX_VARIANTS_TOTAL:
                    break
            if total_possible > MAX_VARIANTS_TOTAL:
                choices = []
                for pos, length, ch in target_runs:
                    opts = [length]
                    if length > 1:
                        opts.append(1)
                    choices.append(opts)
            seen = set()
            for chosen in itertools.product(*choices):
                w = list(token)
                for (pos, orig_len, ch), keep_len in sorted(zip(target_runs, chosen), key=lambda x: x[0][0], reverse=True):
                    del w[pos:pos+orig_len]
                    w[pos:pos] = [ch] * keep_len
                candidate = "".join(w)
                if candidate in seen:
                    continue
                seen.add(candidate)
                cand_norm = candidate
                if normalize_homoglyphs:
                    cand_norm = normalize_confusables(cand_norm)
                cand_norm = unicodedata.normalize("NFKC", cand_norm).casefold()
                for original_token, tok_norm in norm_customs:
                    if cand_norm == tok_norm or collapse_repeats_to_one(cand_norm) == collapse_repeats_to_one(tok_norm):
                        matched_terms.append(normalize_lookup_word(original_token))
                        raw_matched_terms.append((token, "custom_prepass"))
                        custom_matched.append(original_token)
                        return censor_word(token, style)
            return token

        out = token_re.sub(_check_and_censor_token_for_custom, out)

        # 4b) Next: permissive regex for custom tokens (handles leet / separators / homoglyphs)
        custom_patterns = []
        def build_permissive_pattern_for_custom(word: str, max_repeat: int = 5) -> str:
            parts = []
            for ch in word:
                cls = char_class_for(ch)
                parts.append(f"{cls}{{1,{max_repeat}}}")
                parts.append(r"[\W_]{0,3}")
            inner = "".join(parts[:-1])
            return r"(?<!\w)" + inner + r"(?!\w)"

        for original_token, tok_norm in norm_customs:
            try:
                pat_re = re.compile(build_permissive_pattern_for_custom(tok_norm, max_repeat=5), flags=re.IGNORECASE)
            except re.error:
                pat_re = re.compile(re.escape(tok_norm), flags=re.IGNORECASE)
            custom_patterns.append((original_token, tok_norm, pat_re))

        for original_token, variant_used, pattern in custom_patterns:
            def _cust_repl(m):
                raw = m.group(0)
                matched_terms.append(normalize_lookup_word(raw))
                raw_matched_terms.append((raw, "custom_regex"))
                return censor_word(raw, style)
            out_before = out
            out = pattern.sub(_cust_repl, out)
            if out != out_before:
                custom_matched.append(original_token)

        if custom_matched:
            reasons.append("custom_blocklist")
            for cm in custom_matched:
                matched_terms.append(normalize_lookup_word(cm))

    # 5) Main fuzzy regex pass (collect matches)
    patterns = PRECOMPILED.get(mode, [])
    for regex in patterns:
        def repl(m):
            raw = m.group(0)
            matched_terms.append(normalize_lookup_word(raw))
            raw_matched_terms.append((raw, "regex"))
            return censor_word(raw, style)
        out = regex.sub(repl, out)

    # 6) Final sanitize (failsafe)
    out = final_sanitize(out, style, mode, matched_terms)

    # 7) If we detected matched terms from any pass, ensure 'profanity' is present
    if matched_terms:
        if "profanity" not in reasons:
            reasons.append("profanity")

    # 8) Decide blocked flag and suggested action
    blocked_flag = (out != original_text) or bool(custom_matched) or ("emoji_middle_finger" in reasons)
    severity = 0.0
    if "stretched_variant" in reasons:
        severity = max(severity, 0.5)
    if "custom_blocklist" in reasons:
        severity = max(severity, 0.6)
    normalized_matches = [m for m in matched_terms]
    if any(m in BAD_EXTREME_LOOKUP or collapse_repeats_to_one(m) in BAD_EXTREME_COLLAPSED for m in normalized_matches):
        severity = max(severity, 0.95)

    action_suggested = "allow"
    if severity >= 0.75:
        action_suggested = "hard_block"
    elif severity >= 0.45:
        action_suggested = "soft_block"
    elif blocked_flag:
        action_suggested = "soft_block"

    matched_terms = list(dict.fromkeys(matched_terms))
    reasons = list(dict.fromkeys(reasons))

    out = re.sub(r'\s{2,}', ' ', out).strip()

    response = {
        "original_text": original_text,
        "normalized_text": maybe_normalized,
        "clean_text": out,
        "mode": mode,
        "blocked": bool(blocked_flag),
        "action_suggested": action_suggested,
        "severity": round(float(severity), 3),
        "reasons": reasons,
        "matched_terms": matched_terms,
        "truncated": truncated,
    }
    return response

# ---------------------------
# LRU cache wrapper
# ---------------------------
@lru_cache(maxsize=8192)
def _censor_cache_key(key: str) -> str:
    return key

def censor_cached(original_text: str, style: str, mode: str, normalize_homoglyphs: bool, custom_blocklist: Optional[List[str]]):
    cb_key = ",".join(sorted([t.casefold() for t in (custom_blocklist or [])])) if custom_blocklist else ""
    key = f"{mode}|{style}|{int(normalize_homoglyphs)}|{cb_key}|{original_text}"
    res = _censor_cached_call(key, original_text, style, mode, normalize_homoglyphs, cb_key)
    return res

@lru_cache(maxsize=8192)
def _censor_cached_call(serialized_key: str, original_text: str, style: str, mode: str, normalize_homoglyphs: bool, cb_key: str):
    return smart_censor_unwrapped(original_text, style, mode, normalize_homoglyphs, cb_key.split(",") if cb_key else None)

# ---------------------------
# Async endpoints: use run_in_executor to avoid blocking event loop
# ---------------------------
async def run_censor_in_executor(original_text: str, style: str, mode: str,
                                 normalize_homoglyphs: bool, custom_blocklist: Optional[List[str]],
                                 timeout: float = 1.5):
    loop = asyncio.get_running_loop()
    fn = partial(censor_cached, original_text, style, mode, normalize_homoglyphs, custom_blocklist)
    try:
        task = loop.run_in_executor(None, fn)
        result = await asyncio.wait_for(task, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        return {
            "original_text": original_text,
            "normalized_text": "",
            "clean_text": "",
            "mode": mode,
            "blocked": True,
            "action_suggested": "review",
            "severity": 0.9,
            "reasons": ["processing_timeout"],
            "matched_terms": [],
            "truncated": False,
        }

@app.post("/v1/censor")
async def censor(req: CensorIn, request: Request):
    mode = (req.mode or "all").lower()
    if mode not in {"all", "extreme"}:
        raise HTTPException(status_code=400, detail="mode must be 'all' or 'extreme'" )

    if not req.text or len(req.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    if len(req.text) > 20000:
        raise HTTPException(status_code=413, detail="text too large")

    result = await run_censor_in_executor(req.text, req.style or "stars", mode,
                                         bool(req.normalize_homoglyphs),
                                         req.custom_blocklist or None,
                                         timeout=2.0)

    if not req.metadata:
        return {"clean_text": result.get("clean_text", ""), "mode": mode}

    return result

@app.post("/v1/censor/batch")
async def censor_batch(req: BatchIn, request: Request):
    if not req.items:
        raise HTTPException(status_code=400, detail="items must be a non-empty list")
    out = []
    for item in req.items:
        res = await run_censor_in_executor(item.text, item.style or "stars", (item.mode or "all").lower(),
                                           bool(item.normalize_homoglyphs),
                                           item.custom_blocklist or None,
                                           timeout=2.0)
        out.append(res)
    return {"items": out}

# ---------------------------
# If you want to run locally:
# uvicorn main:app --reload
# ---------------------------
