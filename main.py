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
MAX_VARIANTS_TOTAL = 50
VOWELS = set("aeiouAEIOU")

def find_consecutive_repeats(word: str):
    """
    Return list of (start_index, length) for runs where a character repeats > 2 times consecutively.
    """
    stretches = []
    i = 0
    while i < len(word):
        j = i + 1
        while j < len(word) and word[j].lower() == word[i].lower():
            j += 1
        length = j - i
        if length > 2:
            stretches.append((i, length))
        i = j
    return stretches

def is_vowel(ch: str):
    return ch.lower() in VOWELS

def is_consonant(ch: str):
    return ch.isalpha() and ch.lower() not in VOWELS

def generate_reduction_variants(original: str, cap_per_block=MAX_REDUCTION_PER_BLOCK, max_total=MAX_VARIANTS_TOTAL):
    """
    Given an original token (string), find repeated blocks (vowels or consonants) and create variants
    by reducing each block's length stepwise. Return a list of unique variants (strings).
    Cap total variants to max_total.
    """
    stretches = find_consecutive_repeats(original)
    if not stretches:
        return []

    # keep only stretches (vowels or consonants) but ignore non-alpha repeats
    stretches = [(pos, length) for (pos, length) in stretches if original[pos].isalpha()]
    if not stretches:
        return []

    # Build choices per block: for consonants reduce to (orig-1 ... orig-cap) keeping at least 1;
    # for vowels reduce to 2 and 1 (we prefer reducing vowels to 2 or 1 to catch stretched vowels)
    choices_per_block = []
    for (pos, length) in stretches:
        ch = original[pos]
        if is_consonant(ch):
            max_reduce = min(length - 1, cap_per_block)
            keep_lengths = [length - r for r in range(1, max_reduce + 1)]
            # ensure at least one option
            if not keep_lengths:
                keep_lengths = [length]
            choices_per_block.append(keep_lengths)
        elif is_vowel(ch):
            # for vowels prefer reduce to 2 then to 1 (if possible)
            opts = []
            if length > 2:
                # reduce down to 2
                opts.append(2)
                # also allow 1 if original length > 3 (optional)
                if length > 3:
                    opts.append(1)
            else:
                opts.append(length)
            choices_per_block.append(opts)
        else:
            choices_per_block.append([length])

    variants = []
    seen = set()

    # Cartesian product of choices (cap total variants)
    for chosen in itertools.product(*choices_per_block):
        w = list(original)
        # apply from right to left so positions don't shift
        for (pos, orig_len), keep_len in sorted(zip(stretches, chosen), key=lambda x: x[0][0], reverse=True):
            ch = w[pos]
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
    Walk tokens, find tokens with letter runs >2 (vowel or consonant),
    generate variants (reducing stretches) and if any variant matches badlist, mask original token.
    """
    token_re = re.compile(r"\b[\w@#\$%!\|\-']+\b", flags=re.UNICODE)

    def replace_token(m):
        token = m.group(0)
        stretches = find_consecutive_repeats(token)
        if not stretches:
            return token
        # only proceed if there's at least one alphabetic repeated block
        if not any(token[pos].isalpha() for pos, _ in stretches):
            return token
        # generate variants
        variants = generate_reduction_variants(token)
        for v in variants:
            if variant_matches_lookup(v, mode):
                return censor_word(token, style)
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
