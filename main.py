# main.py
import os
import re
import random
import pathlib
import itertools
import unicodedata
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Profanity Filter API (multilingual + stretch handling + normalizer)")

# Backend secret for RapidAPI/Render
BACKEND_SECRET: Optional[str] = os.getenv("BACKEND_SECRET")

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

class CensorIn(BaseModel):
    text: str
    style: Optional[str] = "stars"    # "stars" | "symbols" | "mask" | "blocks" | "red"
    mode: Optional[str] = "all"      # "all" or "extreme"

@app.get("/health")
def health():
    return {"status": "ok"}

def load_wordfile(name: str, comma_separated: bool = False):
    """Load words from a file. Supports newline or comma separated."""
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

# ---------------------------
# Load lists
# ---------------------------
BAD_ALL = load_wordfile("badwords_all.txt", comma_separated=False)       # English general
BAD_EXTREME = load_wordfile("badwords_extreme.txt", comma_separated=True) # English extreme (comma sep)
BAD_MISC = load_wordfile("badwords_misc.txt", comma_separated=False)      # Multilingual (Spanish, Portuguese, French)

# Merge misc words into all sets
BAD_ALL |= BAD_MISC
BAD_EXTREME |= BAD_MISC

# Fallback if files are missing
if not BAD_ALL:
    BAD_ALL = {"shit","fuck","bitch","idiot","loser","simp","gooner"}
if not BAD_EXTREME:
    BAD_EXTREME = {"kys","rape","nigger","faggot"}
if not BAD_MISC:
    BAD_MISC = {"puta","merda","merde","cabron","connard"}

# Build normalized lookup sets for quick membership checks
def normalize_lookup_word(w: str) -> str:
    # strip diacritics and lowercase
    nw = unicodedata.normalize("NFKD", w)
    nw = "".join(ch for ch in nw if not unicodedata.combining(ch))
    return nw.lower()

# helper: collapse repeated letters to a single char for lookup matching
def collapse_repeats_to_one(s: str) -> str:
    return re.sub(r'(.)\1+', r'\1', s, flags=re.IGNORECASE)

BAD_ALL_LOOKUP = {normalize_lookup_word(w) for w in BAD_ALL}
BAD_EXTREME_LOOKUP = {normalize_lookup_word(w) for w in BAD_EXTREME}
# also prepare collapsed forms set for fuzzy equality checks
BAD_ALL_COLLAPSED = {collapse_repeats_to_one(normalize_lookup_word(w)) for w in BAD_ALL}
BAD_EXTREME_COLLAPSED = {collapse_repeats_to_one(normalize_lookup_word(w)) for w in BAD_EXTREME}

SYMBOLS = ["@", "#", "$", "!", "%", "&"]

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
# Leetspeak substitution map
# ---------------------------
LEET_MAP = {
    "a": ["a", "@", "4"],
    "b": ["b", "8"],
    "c": ["c", "(" , "<"],
    "d": ["d"],
    "e": ["e", "3"],
    "f": ["f"],
    "g": ["g", "9"],
    "h": ["h", "#"],
    "i": ["i", "1", "!", "|"],
    "j": ["j"],
    "k": ["k"],
    "l": ["l", "1", "|"],
    "m": ["m"],
    "n": ["n"],
    "o": ["o", "0"],
    "p": ["p"],
    "q": ["q"],
    "r": ["r"],
    "s": ["s", "$", "5"],
    "t": ["t", "7", "+"],
    "u": ["u", "v", "@"],
    "v": ["v", "\\/"],
    "w": ["w"],
    "x": ["x", "%"],
    "y": ["y"],
    "z": ["z", "2"],
}

def char_class_for(ch: str) -> str:
    ch_low = ch.lower()
    if ch_low in LEET_MAP:
        parts = [re.escape(x) for x in set(LEET_MAP[ch_low])]
        # sort longer first to avoid partial matches like "\" vs "\/"
        parts = sorted(parts, key=lambda s: -len(s))
        return "(" + "|".join(parts) + ")"
    else:
        return re.escape(ch)

def build_fuzzy_pattern(word: str) -> str:
    """
    Build a fuzzy regex pattern for `word` that:
      - allows leet substitutions
      - allows up to a few non-alphanumeric separators between letters
      - allows stretched vowels (handled by pattern)
      - allows limited trailing consonant repeats
    """
    tokens = word.split()
    token_patterns = []
    for token in tokens:
        parts = []
        for i, ch in enumerate(token):
            if ch.lower() in "aeiou":
                # vowels: match 1-2 normal + allow extra repeats (regex handles limited stretch)
                parts.append(char_class_for(ch) + "{1,2}")
            else:
                # consonants: normal, but if last char allow small repeat (1-5)
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
# Precompile regex patterns
# ---------------------------
PRECOMPILED = {
    "all": [],
    "extreme": []
}

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
# Helper: remove middle-finger emoji (any skin tone)
# ---------------------------
def remove_middle_fingers(text: str) -> str:
    out_chars = []
    skip_skin_tone = False
    for ch in text:
        if ch == "\U0001F595":  # 🖕
            skip_skin_tone = True
            continue
        if skip_skin_tone and "\U0001F3FB" <= ch <= "\U0001F3FF":
            # skip skin tone if it comes right after 🖕
            skip_skin_tone = False
            continue
        skip_skin_tone = False  # reset after one char
        out_chars.append(ch)
    return "".join(out_chars)


# ---------------------------
# Repeated-stretch helpers (pre-pass)
# ---------------------------
MAX_REDUCTION_PER_BLOCK = 5
MAX_VARIANTS_TOTAL = 200
VOWELS = set("aeiouAEIOU")

def is_vowel(ch: str):
    return ch.lower() in VOWELS

def is_consonant(ch: str):
    return ch.isalpha() and ch.lower() not in VOWELS

def find_alpha_runs(word: str):
    """
    Return list of (start_index, length, char) for consecutive same-char runs (letters only).
    We only return alphabetic runs (letters), skipping digits/punct.
    """
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
    """
    Given an original token (string), find repeated alpha runs and create variants
    by reducing each run's length stepwise. Return a list of unique variants (strings).
    This version accepts runs of length >=2 (not just >2) but we cap choices to avoid explosion.
    """
    runs = find_alpha_runs(original)
    # keep only runs with length >=2 (alpha)
    target_runs = [(pos, length, ch) for (pos, length, ch) in runs if length >= 2]
    if not target_runs:
        return []

    # Build choices per target run
    choices_per_run = []
    for pos, length, ch in target_runs:
        if is_vowel(ch):
            # reduce vowels to 2 and optionally 1
            opts = []
            if length > 2:
                opts.append(2)
                if length > 3:
                    opts.append(1)
            else:
                # if length == 2, we still allow leaving as 2 and also 1 — useful for adj-run cases
                opts = [2, 1]
            choices_per_run.append(sorted(set(opts)))
        elif is_consonant(ch):
            # for consonants allow keeping original length and reducing down to 1 (bounded)
            opts = list(range(max(1, length - cap_per_block), length + 1))
            # ensure descending preference: keep larger lengths first for realistic variants
            choices_per_run.append(sorted(set(opts), reverse=True))
        else:
            choices_per_run.append([length])

    variants = []
    seen = set()
    # Cartesian product but cap total
    for chosen in itertools.product(*choices_per_run):
        w = list(original)
        # apply from right to left so positions don't shift
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
    """
    Check candidate against lookup sets.
    We compare:
     - normalized candidate as-is
     - collapsed candidate (repeated letters -> one) to catch stretched vowels
    """
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

def pre_censor_repeated_stretches(text: str, mode: str, style: str) -> str:
    """
    Simple rule implementation:
      - find tokens containing repeated letters (runs length >= 2)
      - for those tokens, generate variants by reducing each repeated run (trying lengths 1..orig_len capped)
      - if any variant matches the bad-word lookup, censor the ORIGINAL token
      - otherwise keep the original token unchanged
    Safety caps:
      - cap per-run reduction to MAX_REDUCTION_PER_BLOCK (e.g., if run length 10 and cap 5, try keep lengths 1..10 but reduce choices are capped)
      - cap total variants to MAX_VARIANTS_TOTAL
    """
    token_re = re.compile(r"\b[\w@#\$%!\|\-']+\b", flags=re.UNICODE)

    def replace_token(m):
        token = m.group(0)

        # quick guard: don't try tiny tokens
        if len(token) < 3:
            return token

        # find alphabetic runs (letters only)
        runs = find_alpha_runs(token)  # from your helper above
        # filter only runs length >= 2
        target_runs = [(pos, length, ch) for (pos, length, ch) in runs if length >= 2 and ch.isalpha()]
        if not target_runs:
            return token

        # Build choices per run: for each repeated block, choose a keep-length list.
        choices = []
        for pos, length, ch in target_runs:
            # prefer to try realistic reductions:
            # allow keep-length from 1 .. min(length, length) but cap amount of choices
            # we choose few representative options: keep original, keep smaller down to 1 (capped)
            max_keep = length
            min_keep = 1
            # create a bounded list of keep lengths (descending preference)
            # e.g., for length=6 and cap=5 -> opts = [6,5,4,3,2,1] but we'll trim to reasonable size if needed
            opts = list(range(max_keep, min_keep - 1, -1))
            # trim options if too many (limit per block to MAX_REDUCTION_PER_BLOCK choices)
            if len(opts) > MAX_REDUCTION_PER_BLOCK:
                opts = opts[:MAX_REDUCTION_PER_BLOCK]
            choices.append(opts)

        # If product is huge, bail early
        total_possible = 1
        for c in choices:
            total_possible *= len(c)
            if total_possible > MAX_VARIANTS_TOTAL:
                break
        if total_possible > MAX_VARIANTS_TOTAL:
            # fallback: be conservative — only try the most-likely reductions: reduce each run to either original or 1
            choices = []
            for pos, length, ch in target_runs:
                opts = [length]
                if length > 1:
                    opts.append(1)
                choices.append(opts)

        # generate variants (Cartesian product), cap results
        variants_tried = 0
        seen = set()
        for chosen in itertools.product(*choices):
            # construct candidate by applying chosen keep lengths (apply right-to-left so indexes stay valid)
            w = list(token)
            for (pos, orig_len, ch), keep_len in sorted(zip(target_runs, chosen), key=lambda x: x[0][0], reverse=True):
                # remove the original run and insert keep_len copies
                del w[pos:pos+orig_len]
                w[pos:pos] = [ch] * keep_len
            candidate = "".join(w)
            if candidate in seen:
                continue
            seen.add(candidate)
            variants_tried += 1
            # check lookup
            if variant_matches_lookup(candidate, mode):  # your helper above
                return censor_word(token, style)
            if variants_tried >= MAX_VARIANTS_TOTAL:
                break

        # no matching variant found -> return original unchanged
        return token

    return token_re.sub(replace_token, text)



# ---------------------------
# Final sanitize: run PRECOMPILED regexes once more to be safe
# ---------------------------
def final_sanitize(text: str, style: str, mode: str) -> str:
    patterns = PRECOMPILED.get(mode, [])
    for regex in patterns:
        text = regex.sub(lambda m: censor_word(m.group(0), style), text)
    return text

# ---------------------------
# Main censor logic (new flow)
# ---------------------------
def smart_censor(text: str, style: str, mode: str) -> str:
    # 1) operate on original but remove middle-finger emoji immediately
    clean = remove_middle_fingers(text)

    # 2) pre-pass: mask tokens with suspicious letter stretches if variants match badlist
    clean = pre_censor_repeated_stretches(clean, mode, style)

    # 3) main fuzzy regex pass (leetspeak, separators, stretched vowels etc.)
    patterns = PRECOMPILED.get(mode, [])
    for regex in patterns:
        clean = regex.sub(lambda m: censor_word(m.group(0), style), clean)

    # 4) final sanitize pass (failsafe)
    clean = final_sanitize(clean, style, mode)

    return clean

@app.post("/v1/censor")
def censor(req: CensorIn):
    mode = (req.mode or "all").lower()
    if mode not in {"all", "extreme"}:
        raise HTTPException(status_code=400, detail="mode must be 'all' or 'extreme'" )
    clean = smart_censor(req.text, req.style or "stars", mode)
    return {"clean_text": clean, "mode": mode}
