import os
import re
import random
import pathlib
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
    "u": ["u", "v", "@"],   # include "@" so f@ck matches
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
                # vowels: match 1-2 normal + allow extra repeats (we pre-normalize long stretches)
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
    skip_next_ft = False
    for ch in text:
        if ch == "\U0001F595":  # 🖕
            skip_next_ft = True
            continue
        if skip_next_ft and "\U0001F3FB" <= ch <= "\U0001F3FF":
            skip_next_ft = False
            continue
        if skip_next_ft:
            skip_next_ft = False
        else:
            out_chars.append(ch)
    return "".join(out_chars)

# ---------------------------
# Normalizer: collapse suspicious letter stretches
# ---------------------------
VOWEL_CLASS = "aeiou"
# collapse 3+ vowels -> 2 vowels; collapse 3+ consonants -> 1 consonant
def normalize_stretch(text: str) -> str:
    # 1) collapse vowels repeated 3+ to exactly 2 (case-insensitive)
    text = re.sub(r'([aeiouAEIOU])\1{2,}', lambda m: m.group(1) * 2, text)
    # 2) collapse consonants repeated 3+ to single occurrence (avoid matching vowels)
    text = re.sub(r'([^aeiouAEIOU\W_])\1{2,}', lambda m: m.group(1), text)
    return text

# ---------------------------
# Main censor logic
# ---------------------------
def smart_censor(text: str, style: str, mode: str) -> str:
    # normalize suspicious stretches first to improve matching speed & avoid false positives
    normalized = normalize_stretch(text)

    censored = normalized
    patterns = PRECOMPILED.get(mode, [])
    for regex in patterns:
        censored = regex.sub(lambda m: censor_word(m.group(), style), censored)

    censored = remove_middle_fingers(censored)
    return censored

@app.post("/v1/censor")
def censor(req: CensorIn):
    mode = (req.mode or "all").lower()
    if mode not in {"all", "extreme"}:
        raise HTTPException(status_code=400, detail="mode must be 'all' or 'extreme'" )
    clean = smart_censor(req.text, req.style or "stars", mode)
    return {"clean_text": clean, "mode": mode}
