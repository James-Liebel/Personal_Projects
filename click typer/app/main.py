# --- Click Typer main script (Gemini sentence mutation + strict output fidelity + human typing realism) ---

import sys, pathlib, os, re, random, time, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.auth.google_auth import get_docs_client
from app.docloc.insertion_point import resolve_insertion_point
from app.humantyping.human_typer import _replace_all_text, _batch_insert
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type

# =============================
# 🧠 CONFIG — edit these freely
# =============================

DOC_ID = "14avfHf5sHoBJLbQRuHHrTygqwno1dNPAOvGNhIAD_A8"
MARKER = "     "
WPM = 85

# Typing realism
TYPO_PROB = 0.1
RANDOM_PAUSE_PROB = 0.10
THINK_TIME_RANGE = (0.1, 0.4)
WORDS_PER_BATCH_LIMIT = 5
CHARS_PER_BATCH_LIMIT = 22

# Strictly correct, non-typo text
TEXT_TO_TYPE = (

"""The Drum Awards for Marketing (Americas) 2025 highlighted ten major trends shaping the industry’s evolution across creativity, technology, and culture.

1. **Cultural agility** – Brands like CarBravo succeeded by tapping into cultural moments, proving that timely, value-driven campaigns resonate deeply.
2. **AI-driven personalization** – Headspace showed how AI can create emotionally intelligent personalization at scale while maintaining brand sensitivity.
3. **Experience as brand equity** – Ford’s revival of Michigan Central Station demonstrated how live experiences can strengthen brand identity and public perception.
4. **Emotion in B2B** – Mack Trucks proved that emotional storytelling can boost engagement and sales even in industrial sectors.
5. **Insight-led innovation** – Hilton turned consumer data into actionable brand assets that fueled media, product, and PR success.
6. **Gaming as audience gateway** – Coldplay’s Roblox campaign illustrated how immersive gaming connects brands with younger audiences.
7. **Tech transforming traditional media** – NBCUniversal’s real-time data visualization redefined audience engagement and news storytelling.
8. **Purpose-led cultural messaging** – Elf Beauty’s “Change the Board Game” campaign used activism to drive awareness and real corporate change.
9. **Cross-platform fan ecosystems** – Integrating content across multiple channels, like Coldplay’s campaign, built stronger and broader fan engagement.
10. **Empathetic storytelling** – Fuck Cancer’s honest approach to health communication showed the power of raw, human-centered narratives.

Overall, the 2025 winners revealed that modern marketing isn’t just about selling—it’s about moving culture, building community, and humanizing brands through creativity and authenticity.
"""

)

# =====================================
# 🤖 Gemini AI Setup (robust, no *-exp, auto-fallback on 429)
# =====================================

AI_MUTATE = False
gemini_mutate = None

def _setup_gemini():
    try:
        import google.generativeai as genai
    except ImportError:
        print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")
        return None, False

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ GEMINI_API_KEY not found — AI mutation disabled.")
        return None, False

    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"⚠️ Gemini configure failed: {e}")
        return None, False

    CANDIDATES = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    def _pick_model():
        try:
            avail = list(genai.list_models())
            names = []
            for m in avail:
                if "generateContent" in getattr(m, "supported_generation_methods", []):
                    name = m.name.split("/")[-1]
                    if not name.endswith("-exp"):
                        names.append(name)
            for cand in CANDIDATES:
                if cand in names:
                    return cand
            return names[0] if names else CANDIDATES[0]
        except Exception:
            return CANDIDATES[0]

    model_name = _pick_model()
    try:
        generation_config = {"max_output_tokens": 60, "temperature": 0.9}
        model = genai.GenerativeModel(model_name, generation_config=generation_config)
        print(f"✅ Gemini sentence mutation enabled (model: {model_name}).")
    except Exception as e:
        print(f"⚠️ Gemini model init failed ({model_name}): {e}")
        return None, False

    state = {"disabled": False}

    def _mutate(sentence: str) -> str:
        if state["disabled"]:
            return sentence
        prompt = (
            "Rephrase the following sentence using different words and phrasing while "
            "preserving all original punctuation (commas, periods, quotes, etc.), in exactly the same order and position. "
            "Do not add or remove any punctuation or change their order. Only change words. "
            "If you cannot rephrase without modifying punctuation, return the original.\n"
            f"Sentence: {sentence}\nRephrased:"
        )
        try:
            resp = model.generate_content([prompt])
            text = getattr(resp, "text", None)
            if not text and getattr(resp, "candidates", None):
                parts = resp.candidates[0].content.parts
                if parts and hasattr(parts[0], "text"):
                    text = parts[0].text
            out = (text or "").strip().strip('"').strip("'")
            # Always double check: strict output match
            if not out or out == sentence:
                return sentence
            orig_punct = [(i, c) for i, c in enumerate(sentence) if c in '.!?,:;\'"()-']
            new_punct = [(i, c) for i, c in enumerate(out) if c in '.!?,:;\'"()-']
            if (len(orig_punct) == len(new_punct) and
                [c for (_, c) in orig_punct] == [c for (_, c) in new_punct]):
                return out
            return sentence
        except Exception as e:
            msg = str(e)
            if "429" in msg or "quota" in msg.lower() or "rate limit" in msg.lower():
                print("⚠️ Gemini quota/rate-limit detected — disabling AI mutation for this run.")
                state["disabled"] = True
                return sentence
            print(f"⚠️ Gemini mutation failed: {e}")
            return sentence

    return _mutate, True

gemini_mutate, AI_MUTATE = _setup_gemini()

# =============================
# --- Fake edits configuration ---
# =============================
FAKE_EDIT_PROB = 0.10          # dialed down to reduce write bursts
FAKE_EDIT_TYPES = {            # weights for which edit to perform
    "word": 0.55,              # delete last word(s) and retype
    "fidget": 0.30,            # small deletion, type junk, delete, then correct
    "sentence": 0.15           # delete everything typed-so-far (this sentence) and retype that exact span
}
MAX_WORDS_BACK = 2             # how many words to delete on a 'word' edit
MAX_FIDGET_CHARS = 2           # small, realistic blips to ease quota
SENTENCE_EDIT_MIN_CHARS = 18   # only consider sentence retype when we’ve typed at least this many chars

# =============================
# Utilities
# =============================

def calc_delays_from_wpm(wpm: int):
    cps = max(wpm * 5.0 / 60.0, 1.0)
    min_ms = max(1000.0 / cps, 45.0)
    max_ms = min_ms * 3.0
    return min_ms / 1000.0, max_ms / 1000.0

def tokens_of(s: str):
    token_re = re.compile(r'\S+|\s+')
    return [m.group() for m in token_re.finditer(s)]

KEYBOARD_MAP = {
    'q': ['w', 'a'], 'w': ['q', 'e', 'a', 's'], 'e': ['w', 'r', 's', 'd'],
    'r': ['e', 't', 'd', 'f'], 't': ['r', 'y', 'f', 'g'], 'y': ['t', 'u', 'g', 'h'],
    'u': ['y', 'i', 'h', 'j'], 'i': ['u', 'o', 'j', 'k'], 'o': ['i', 'p', 'k', 'l'],
    'p': ['o', 'l'], 'a': ['q', 'w', 's', 'z'], 's': ['a', 'w', 'e', 'd', 'z', 'x'],
    'd': ['s', 'e', 'r', 'f', 'x', 'c'], 'f': ['d', 'r', 't', 'g', 'c', 'v'],
    'g': ['f', 't', 'y', 'h', 'v', 'b'], 'h': ['g', 'y', 'u', 'j', 'b', 'n'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'], 'z': ['a', 's', 'x'], 'x': ['z', 's', 'd', 'c'],
    'c': ['x', 'd', 'f', 'v'], 'v': ['c', 'f', 'g', 'b'], 'b': ['v', 'g', 'h', 'n'],
    'n': ['b', 'h', 'j', 'm'], 'm': ['n', 'j', 'k', 'l']
}

def nearby_key(ch: str) -> str:
    lo = ch.lower()
    if lo in KEYBOARD_MAP:
        rep = random.choice(KEYBOARD_MAP[lo])
        return rep.upper() if ch.isupper() else rep
    return ch

def maybe_typo(token: str) -> str:
    # Never actually emit a typo; always return original token.
    return token

def typo_and_correction_sequence(word: str):
    # ALWAYS emit only the correct word and no correction, never emit an error!
    return [(word, None)]

@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), reraise=True)
def _batch_insert_safe(docs_client, doc_id, start_index, text):
    _batch_insert(docs_client, doc_id, start_index, text)

# Retry delete with exponential backoff on rate limits
def _delete_request(docs_client, doc_id, start_index, num_chars):
    requests = [{
        "deleteContentRange": {
            "range": {"startIndex": start_index, "endIndex": start_index + num_chars}
        }
    }]
    docs_client.documents().batchUpdate(
        documentId=doc_id, body={"requests": requests}
    ).execute()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=1, max=6),
    retry=retry_if_exception_type(HttpError),
    reraise=True
)
def _batch_delete_safe(docs_client, doc_id, start_index, num_chars):
    try:
        _delete_request(docs_client, doc_id, start_index, num_chars)
    except HttpError as e:
        # Bubble up for tenacity to backoff/retry on 429s
        raise
    except Exception as e:
        print(f"⚠️ Backspace simulation failed at {start_index} for {num_chars} chars: {e}")
        raise

def split_word_realistic(word):
    # Realistically split a long word for human-like chunking.
    if len(word) < 6:
        return [word]
    parts = []
    ptr = 0
    while ptr < len(word):
        remain = len(word) - ptr
        if remain <= 3:
            chunklen = remain
        else:
            chunklen = random.choices([2,3,4], weights=[0.5,0.35,0.15])[0]
            if chunklen > remain:
                chunklen = remain
        parts.append(word[ptr:ptr+chunklen])
        ptr += chunklen
    return parts

def type_token_in_chunks(docs_client, doc_id, idx:int, token:str, min_delay, max_delay):
    if not token.strip():
        _batch_insert_safe(docs_client, doc_id, idx, token)
        idx += len(token)
        time.sleep(random.uniform(min_delay, max_delay))
        return idx
    subchunks = split_word_realistic(token)
    for part in subchunks:
        _batch_insert_safe(docs_client, doc_id, idx, part)
        idx += len(part)
        delay = random.uniform(min_delay, max_delay)
        if random.random() < RANDOM_PAUSE_PROB:
            delay += random.uniform(*THINK_TIME_RANGE)
        time.sleep(delay)
    return idx

# -----------------------------
# Junk generation (realistic nearby keys / small word variants)
# -----------------------------

def generate_junk_like(context_tail: str, max_len: int) -> str:
    """
    Produce a tiny burst of junk chars that feels like real mistakes:
    - Prefer nearby-key substitutions of the last few visible chars.
    - Occasionally duplicate a character (e.g., 'goa' -> 'goaa').
    """
    # pull a small tail of context to perturb
    base = "".join([c for c in context_tail[-6:] if not c.isspace()]) or "the"
    base = base[-max_len:] if len(base) > max_len else base
    out = []
    for ch in base:
        if ch.isalpha() and random.random() < 0.8:
            out.append(nearby_key(ch))
        else:
            out.append(ch)
        # occasional accidental repeat
        if random.random() < 0.15:
            out.append(out[-1])
        if len(out) >= max_len:
            break
    if not out:
        # fall back: quick letters from the home row shaped by nearby keys
        seed = "asdfjkl"
        out = [random.choice(seed)]
        while len(out) < max_len:
            out.append(nearby_key(out[-1]))
    return "".join(out)[:max_len]

# -----------------------------
# Fake edit helpers and engine
# -----------------------------

def _random_sleep(min_delay, max_delay, think=False):
    delay = random.uniform(min_delay, max_delay)
    if think:
        delay += random.uniform(*THINK_TIME_RANGE)
    time.sleep(delay)

def _choose_weighted(d):
    r = random.random() * sum(d.values())
    cum = 0.0
    for k, w in d.items():
        cum += w
        if r <= cum:
            return k
    return next(iter(d))  # fallback

def _last_n_words_span_len(typed_tokens, n_words):
    """
    Delete up to n_words from the end, respecting whitespace boundaries.
    Returns (span_len, joined_str).
    """
    if not typed_tokens:
        return 0, ""
    i = len(typed_tokens) - 1
    while i >= 0 and typed_tokens[i].isspace():
        i -= 1
    if i < 0:
        return 0, ""
    collected = []
    words_found = 0
    in_word = False
    while i >= 0:
        t = typed_tokens[i]
        collected.append(t)
        if t.isspace():
            if in_word:
                words_found += 1
                in_word = False
                if words_found >= n_words:
                    break
        else:
            in_word = True
        i -= 1
    collected.reverse()
    s = "".join(collected).rstrip()
    return len(s), s

def maybe_fake_edit(
    docs_client,
    doc_id,
    idx,
    out_text,
    sentence_full,
    typed_tokens,            # list of exact tokens typed so far for THIS sentence
    min_delay,
    max_delay,
    allow_edits=True
):
    """
    Randomly performs one of:
      - 'word': delete last 1..MAX_WORDS_BACK words and retype them correctly
      - 'fidget': delete a few chars, type junk, delete junk, then retype correct chars
      - 'sentence': delete everything typed so far in THIS sentence and retype exactly that span
    Returns (idx, out_text, typed_tokens) after the operation (or unchanged).
    """
    if not allow_edits or random.random() > FAKE_EDIT_PROB:
        return idx, out_text, typed_tokens

    edit_kind = _choose_weighted(FAKE_EDIT_TYPES)

    # --- WORD EDIT ---
    if edit_kind == "word" and typed_tokens:
        n = random.randint(1, MAX_WORDS_BACK)
        span_len, span_str = _last_n_words_span_len(typed_tokens, n)
        if span_len <= 0 or span_len > len(out_text):
            return idx, out_text, typed_tokens

        # SAFETY GUARD: ensure exact tail match to avoid duplication
        candidate = span_str
        if not out_text.endswith(candidate):
            if out_text.endswith(" " + candidate):
                candidate = " " + candidate
            else:
                return idx, out_text, typed_tokens
        span_len = len(candidate)
        span_str = candidate

        # delete
        _batch_delete_safe(docs_client, doc_id, idx - span_len, span_len)
        idx -= span_len
        out_text = out_text[:-span_len]

        # trim typed_tokens by chars
        deleted_chars = span_len
        while deleted_chars > 0 and typed_tokens:
            last = typed_tokens.pop()
            l = len(last)
            if l <= deleted_chars:
                deleted_chars -= l
            else:
                remain = last[:-deleted_chars]
                if remain:
                    typed_tokens.append(remain)
                deleted_chars = 0

        _random_sleep(min_delay, max_delay, think=True)

        # retype correct span
        for ch in span_str:
            _batch_insert_safe(docs_client, doc_id, idx, ch)
            idx += 1
            out_text += ch
            typed_tokens.append(ch)
            _random_sleep(min_delay, max_delay)
        return idx, out_text, typed_tokens

    # --- FIDGET EDIT ---
    if edit_kind == "fidget" and len(out_text) >= 2:
        del_count = random.randint(1, min(MAX_FIDGET_CHARS, len(out_text)))
        _batch_delete_safe(docs_client, doc_id, idx - del_count, del_count)
        idx -= del_count
        deleted_chunk = out_text[-del_count:]
        out_text = out_text[:-del_count]

        # trim typed_tokens by del_count chars
        left = del_count
        while left > 0 and typed_tokens:
            t = typed_tokens.pop()
            if len(t) <= left:
                left -= len(t)
            else:
                remain = t[:-left]
                if remain:
                    typed_tokens.append(remain)
                left = 0

        _random_sleep(min_delay, max_delay)

        # junk that looks like real nearby-key slips
        junk_len = random.randint(1, MAX_FIDGET_CHARS)
        junk_seq = generate_junk_like(out_text, junk_len) if out_text else generate_junk_like("the", junk_len)
        for ch in junk_seq:
            _batch_insert_safe(docs_client, doc_id, idx, ch)
            idx += 1
            typed_tokens.append(ch)
            out_text += ch
            _random_sleep(min_delay, max_delay)

        _random_sleep(min_delay, max_delay, think=True)

        # delete junk
        _batch_delete_safe(docs_client, doc_id, idx - junk_len, junk_len)
        idx -= junk_len
        out_text = out_text[:-junk_len]
        for _ in range(junk_len):
            if typed_tokens:
                _ = typed_tokens.pop()

        _random_sleep(min_delay, max_delay)

        # retype the correct deleted chunk
        for ch in deleted_chunk:
            _batch_insert_safe(docs_client, doc_id, idx, ch)
            idx += 1
            out_text += ch
            typed_tokens.append(ch)
            _random_sleep(min_delay, max_delay)
        return idx, out_text, typed_tokens

    # --- SENTENCE EDIT (retype only what's been typed so far) ---
    if edit_kind == "sentence" and len(out_text) >= SENTENCE_EDIT_MIN_CHARS:
        span_len = len(out_text)
        snapshot = out_text  # exactly what was typed so far (includes spaces/punct)
        _batch_delete_safe(docs_client, doc_id, idx - span_len, span_len)
        idx -= span_len
        out_text = ""
        typed_tokens[:] = []
        _random_sleep(min_delay, max_delay, think=True)

        # retype the exact same span (preserves spacing after periods)
        for ch in snapshot:
            _batch_insert_safe(docs_client, doc_id, idx, ch)
            idx += 1
            out_text += ch
            typed_tokens.append(ch)
            _random_sleep(min_delay, max_delay)
        return idx, out_text, typed_tokens

    return idx, out_text, typed_tokens

# -----------------------------
# Typing routines
# -----------------------------

def type_sentence(docs_client, doc_id, idx: int, sentence: str) -> int:
    min_delay, max_delay = calc_delays_from_wpm(WPM)
    toks = tokens_of(sentence)
    out_text = ""
    typed_tokens = []  # exact tokens/chars we insert (for boundary-aware deletions)

    for ti, tok in enumerate(toks):
        last_token = (ti == len(toks) - 1)

        # Insert token in realistic sub-chunks
        if not tok.strip():
            idx = type_token_in_chunks(docs_client, doc_id, idx, tok, min_delay, max_delay)
            out_text += tok
            typed_tokens.append(tok)
        else:
            idx = type_token_in_chunks(docs_client, doc_id, idx, tok, min_delay, max_delay)
            out_text += tok
            typed_tokens.append(tok)

        # Maybe perform a fake edit after each token (disabled on the final token)
        idx, out_text, typed_tokens = maybe_fake_edit(
            docs_client=docs_client,
            doc_id=doc_id,
            idx=idx,
            out_text=out_text,
            sentence_full=sentence,
            typed_tokens=typed_tokens,
            min_delay=min_delay,
            max_delay=max_delay,
            allow_edits=not last_token
        )

    # Final output check - strict enforcement (should always pass)
    assert out_text == sentence, (
        f"INTERNAL ERROR: Typed text and input sentence disagree!\n"
        f"expected={repr(sentence)}\nproduced={repr(out_text)}"
    )
    time.sleep(random.uniform(0.10, 0.25))
    return idx

def fallback_mutate(sentence: str) -> str:
    # Always return the original input
    return sentence

def mutate_sentence(sentence: str) -> str:
    if AI_MUTATE and callable(gemini_mutate):
        out = gemini_mutate(sentence)
        orig_punct = [(i, c) for i, c in enumerate(sentence) if c in '.!?,:;\'"()-']
        new_punct = [(i, c) for i, c in enumerate(out) if c in '.!?,:;\'"()-'] if out else []
        if out and out != sentence and len(orig_punct) == len(new_punct) and [c for (_, c) in orig_punct] == [c for (_, c) in new_punct]:
            return out
        else:
            return sentence
    return fallback_mutate(sentence)

def smart_typing_with_sentence_mutation(docs_client, doc_id: str, start_index: int, text: str, marker: str):
    _replace_all_text(docs_client, doc_id, marker, "")
    idx = int(start_index)

    # Find sentences with trailing punctuation; also emit the exact gaps between them.
    sentence_re = re.compile(r"[^.!?]+[.!?\"']*", flags=re.MULTILINE)
    matches = list(sentence_re.finditer(text))

    for i, m in enumerate(matches, 1):
        original = m.group()                 # sentence including its punctuation
        mutated = mutate_sentence(original)  # should equal original per guard
        assert mutated == original, (
            f"Sentence mutation altered text; strict fidelity required!\n"
            f"Original: {original}\nMutated: {mutated}"
        )

        print(f"Typing sentence {i}/{len(matches)}...")
        idx = type_sentence(docs_client, doc_id, idx, mutated)

        # After the sentence, type the EXACT whitespace gap present in the source
        # before the next sentence (preserves one-space, two-spaces, newline, etc.).
        if i < len(matches):
            gap = text[m.end():matches[i].start()]
        else:
            gap = text[m.end():]  # any trailing whitespace at end of input

        if gap:
            min_delay, max_delay = calc_delays_from_wpm(WPM)
            idx = type_token_in_chunks(docs_client, doc_id, idx, gap, min_delay, max_delay)

    print(f"✅ Typing complete at ~{WPM} WPM (AI mutate: {'Gemini' if AI_MUTATE else 'none'})")

# =============================
# Entry point
# =============================

def main():
    try:
        docs = get_docs_client()
        loc = resolve_insertion_point(docs, DOC_ID, marker=MARKER)
        print(f"Using marker strategy at index {loc.get('index')}...")
        smart_typing_with_sentence_mutation(docs, DOC_ID, loc["index"], TEXT_TO_TYPE, MARKER)
    except HttpError as e:
        print(f"FATAL: Google API error: {e}")
        raise
    except Exception as e:
        print(f"FATAL: {e}")
        raise

if __name__ == "__main__":
    main()