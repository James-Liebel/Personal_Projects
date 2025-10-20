import time, random, os, pickle, re, string
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ---------------- CONFIG ----------------
SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'

# Typing behavior
WPM = 40
CHAR_JITTER = 0.25
TYPO_CHANCE = 0.10
BACKTRACK_CHANCE = 0.3
QUOTE_DELAY_RANGE = (15, 30)   # seconds
PARA_DELAY_RANGE = (10,20)   # 2.5-5 minutes
PARA_START_RANGE = (0.8, 2.0)  # hesitation at new paragraph starts

# Extra behavior toggles (tune probabilities here)
SENTENCE_REWRITE_PROB = 0.15
DELETE_BURST_PROB = 0.025
MIDSTREAM_FIX_PROB = 0.02
PASTE_TAILOR_PROB = 0.08
SHIFT_FLUB_PROB = 0.03
LATENCY_SPIKE_PROB = 0.01
CLEANUP_PASS_PROB = 0.005

CANNED_SNIPPETS = [
    "In summary, ",
    "More importantly, ",
    "From a strategic standpoint, "
]

# ---------------- INPUT ----------------
TEXT_TO_TYPE = """1. What is Swift's attitude toward the beggars he describes in the opening paragraph?
Swift writes about the beggars in a cold, detached, almost statistical way. He describes them as if they are a nuisance or burden to society rather than as human beings. This deliberately harsh tone sets up the satire—he is mimicking how the ruling classes and policymakers viewed the poor.

2. Where do the speaker's allegiances lie in this essay? With what social groups does he identify himself?
The speaker aligns himself with the wealthy, land-owning, and politically powerful classes (English landlords and the Anglo-Irish elite). He speaks as if he were one of them, showing disdain for the poor while pretending to propose a “solution” for their suffering.

3. What sort of persona does Swift create for the "author" of A Modest Proposal?
The persona is that of a rational, calculating, utilitarian “projector” (someone who proposes economic schemes). This figure is heartless, focused only on numbers, profit, and efficiency, with no moral sensitivity.

4. Where do you detect differences between the "proposer" and Swift himself?
The “proposer” seems completely serious about eating children, whereas Swift himself clearly intends this as satire. Swift’s real voice comes through in the biting irony, in the exaggerated coldness, and in the underlying anger at English exploitation of Ireland and the neglect of the poor.

5. If Swift does not actually think the Irish people should eat their children, what does he think they should do?
Swift’s deeper point is that England should stop oppressing Ireland, landlords should treat tenants fairly, and Irish society should pursue real reforms—such as economic self-reliance, honesty in trade, and compassion toward the poor—rather than cruel or exploitative schemes.

6. Who is the audience of this work?
Primarily, Swift was targeting the English ruling class and wealthy Anglo-Irish landlords. They were the ones in power, responsible for policies that worsened poverty in Ireland. At the same time, Irish readers would recognize the satire and see Swift’s critique of both the English and the complacency of some Irish elites.

7. Who will be the beneficiaries of this "Modest Proposal"?
According to the satirical argument, the landlords and wealthy would benefit most—because they would literally “eat” the children of the poor while making money from the scheme. In truth, Swift is showing how they already “devour” the poor through unfair rents and exploitation.

8. When did it first become apparent to you that Swift's proposal was not serious? How did you respond?
Most readers realize the satire when Swift calmly suggests that infants could be sold as a new form of livestock. The shock of that moment often produces a mix of horror and laughter, making readers question why such an outrageous idea is being treated so logically.

9. What relevance does A Modest Proposal have for contemporary social and political issues? Can you think of historical situations that pose similar problems about ends and means?
The essay remains relevant whenever governments treat human suffering as statistics or seek “efficient” solutions without compassion. Modern examples might include debates over immigration, poverty relief, or healthcare where cost-saving is prioritized over human dignity. Historically, one could compare it to exploitative colonial policies, the industrial revolution’s treatment of workers, or even today’s discussions about automation and inequality."""
# ---------------- AUTH -----------------
def authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as f:
            creds = pickle.load(f)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as f:
            pickle.dump(creds, f)
    return build('docs', 'v1', credentials=creds)

# ---------------- DOC HELPERS ----------------
def build_full_text(service, doc_id):
    doc = service.documents().get(documentId=doc_id).execute()
    content = doc.get('body', {}).get('content', [])
    full = ""
    for el in content:
        if 'paragraph' in el:
            for elem in el['paragraph'].get('elements', []):
                tr = elem.get('textRun')
                if tr and 'content' in tr:
                    full += tr['content']
    end_index = 1
    if content:
        end_index = content[-1].get('endIndex', 1)
    return full, end_index

def safe_insert_at(service, doc_id, text, index=None):
    _, end_index = build_full_text(service, doc_id)
    insert_index = max(1, end_index-1) if index is None else max(1, index)
    req = {"insertText": {"location": {"index": insert_index}, "text": text}}
    service.documents().batchUpdate(documentId=doc_id, body={"requests":[req]}).execute()
    return insert_index + len(text)

def safe_delete_range(service, doc_id, start_index, end_index):
    if start_index >= end_index:
        return
    req = {"deleteContentRange": {"range": {"startIndex": start_index, "endIndex": end_index}}}
    try:
        service.documents().batchUpdate(documentId=doc_id, body={"requests":[req]}).execute()
    except Exception:
        pass

# ---------------- HUMAN BEHAVIOR HELPERS ----------------
def wpm_profile(t_seconds):
    # warm-up → flow → tiny fatigue
    base = WPM
    warmup = min(1.0, t_seconds/60.0)          # ramp first minute
    fatigue = 1.0 - min(0.2, t_seconds/1800.0) # fade over 30 min
    burst = 1.0 + (0.25 if random.random() < 0.05 else 0.0)
    jitter = random.uniform(0.9, 1.1)
    return max(20, base * (0.8 + 0.2*warmup) * fatigue * burst * jitter)

def random_char_delay_dynamic(start_time):
    elapsed = time.time() - start_time
    wpm_now = wpm_profile(elapsed)
    base = 1.0 / ((wpm_now * 5) / 60.0)
    jitter = base * CHAR_JITTER
    return max(0.02, random.gauss(base, jitter))

def cognitive_pause(token, is_sentence_start=False):
    if re.match(r'https?://', token): time.sleep(random.uniform(0.6, 1.2))
    if re.match(r'\d{2,}', token):    time.sleep(random.uniform(0.25, 0.6))
    if len(token) >= 10:              time.sleep(random.uniform(0.15, 0.4))
    if is_sentence_start:             time.sleep(random.uniform(0.3, 0.8))

def punctuation_pause(token):
    if token in {',', ';', ':'}: time.sleep(random.uniform(0.05, 0.2))
    if token in {'.','!','?'}:   time.sleep(random.uniform(0.25, 0.6))
    if token == '—':             time.sleep(random.uniform(0.2, 0.5))

def random_latency_spike():
    if random.random() < LATENCY_SPIKE_PROB:
        time.sleep(random.uniform(3, 8))

def fragment_word(word):
    if len(word) <= 4:
        return [word]
    cut_count = random.choice([1,2])
    cuts = sorted(random.sample(range(1, len(word)), cut_count))
    frags, last = [], 0
    for c in cuts:
        frags.append(word[last:c])
        last = c
    frags.append(word[last:])
    return frags

def shift_flub(token):
    if len(token) > 4 and random.random() < SHIFT_FLUB_PROB:
        i = random.randint(1, len(token)-2)
        return token[:i] + token[i].upper() + token[i+1:]
    return token

def pick_paragraphs(text):
    parts = []
    cur = 0
    paragraphs = text.split("\n\n")
    for p in paragraphs:
        start = text.find(p, cur)
        if start == -1:
            start = cur
        end = start+len(p)
        parts.append((start,end,p))
        cur = end
        if text[cur:cur+2]=="\n\n":
            cur += 2
    return parts

def move_sentence(service, doc_id, source_pars):
    if len(source_pars) < 2: return
    pidx = random.randrange(len(source_pars)-1)
    _,_,para_text = source_pars[pidx]
    sentences = re.split(r'(?<=[.!?])\s+', para_text.strip())
    if len(sentences) < 2: return
    s = random.choice(sentences)
    pos = build_full_text(service, doc_id)[0].rfind(s)
    if pos != -1:
        safe_delete_range(service, doc_id, pos+1, pos+1+len(s))
        insert_at = build_full_text(service, doc_id)[1]//2
        safe_insert_at(service, doc_id, " "+s+" ", insert_at)

def replace_last_word(service, doc_id, correct_word):
    full, _ = build_full_text(service, doc_id)
    m = re.search(r'(\w+)(\W*)$', full)
    if not m: return
    word, trail = m.group(1), m.group(2)
    start = len(full) - len(word+trail) + 1
    end   = start + len(word)
    safe_delete_range(service, doc_id, start, end)
    safe_insert_at(service, doc_id, correct_word, start)

def midstream_fix_prior_word(service, doc_id, pattern=r'\bteh\b', repl='the'):
    full, _ = build_full_text(service, doc_id)
    m = re.search(pattern, full)
    if not m: return
    start = m.start()+1; end = m.end()+1
    safe_delete_range(service, doc_id, start, end)
    safe_insert_at(service, doc_id, repl, start)

def delete_burst(service, doc_id, max_chars=8):
    full, _ = build_full_text(service, doc_id)
    if len(full) <= 2: return
    n = min(len(full)-1, random.randint(2, max_chars))
    start = len(full)-n; end = len(full)-1
    safe_delete_range(service, doc_id, start, end)
    time.sleep(random.uniform(0.1, 0.3))

def rewrite_last_sentence(service, doc_id):
    full, _ = build_full_text(service, doc_id)
    txt = full.strip()
    if not txt: return
    sentences = re.split(r'(?<=[.!?])\s+', txt)
    if not sentences: return
    last = sentences[-1]
    if len(last) < 20 or random.random() > SENTENCE_REWRITE_PROB: return
    start = full.rfind(last) + 1
    safe_delete_range(service, doc_id, start, start+len(last))
    time.sleep(random.uniform(0.5, 1.2))
    variants = [
        re.sub(r'\bvery\b', 'quite', last),
        re.sub(r'\bimportant\b', 'crucial', last),
        re.sub(r', and\b', ' and', last),
        re.sub(r'\s+', ' ', last).strip()
    ]
    safe_insert_at(service, doc_id, random.choice(variants), start)

def cleanup_spacing_caps(service, doc_id):
    full, _ = build_full_text(service, doc_id)
    s = re.sub(r'\.  +', '. ', full)
    s = re.sub(r'(^|[.!?]\s+)([a-z])', lambda m: m.group(1)+m.group(2).upper(), s)
    if s != full:
        safe_delete_range(service, doc_id, 1, len(full))
        safe_insert_at(service, doc_id, s, 1)

def maybe_add_heading(service, doc_id):
    if random.random() < 0.10:
        safe_insert_at(service, doc_id, "\n\n### Notes:\n", 1)

def paste_then_tailor(service, doc_id):
    if random.random() < PASTE_TAILOR_PROB:
        snippet = random.choice(CANNED_SNIPPETS)
        safe_insert_at(service, doc_id, snippet)
        time.sleep(random.uniform(0.2, 0.5))
        if random.random() < 0.5 and snippet.endswith(", "):
            replace_last_word(service, doc_id, snippet[:-2])  # drop comma+space

# ---------------- TYPING PRIMITIVES ----------------
def type_fragment(service, doc_id, frag, start_time):
    safe_insert_at(service, doc_id, frag)
    time.sleep(sum(random_char_delay_dynamic(start_time) for _ in frag))

def type_token_with_typos(service, doc_id, token, start_time, is_sentence_start):
    # pauses based on content
    cognitive_pause(token, is_sentence_start=is_sentence_start)
    punctuation_pause(token)
    random_latency_spike()

    # treat spaces/punct simply
    if not re.match(r'\w+', token):
        type_fragment(service, doc_id, token, start_time)
        return

    # occasional SHIFT flub on the token
    token_to_type = shift_flub(token)

    # maybe typo first, then delete, then correct
    if random.random() < TYPO_CHANCE:
        chars = list(token_to_type)
        typo_type = random.choice(['swap','missing','extra','nearby'])
        if typo_type == 'swap' and len(chars)>1:
            i = random.randint(0,len(chars)-2)
            chars[i],chars[i+1] = chars[i+1],chars[i]
        elif typo_type == 'missing' and len(chars)>1:
            i = random.randint(0,len(chars)-1); chars.pop(i)
        elif typo_type == 'extra':
            i = random.randint(0,len(chars))
            chars.insert(i, random.choice(string.ascii_lowercase))
        elif typo_type == 'nearby':
            i = random.randint(0,len(chars)-1)
            neighbors = {
                'a':'qwsz','b':'vghn','c':'xdfv','d':'ersfcx','e':'wsdr','f':'rtgvcd','g':'tyhbvf',
                'h':'yujnbg','i':'ujko','j':'uikhmn','k':'ijolm','l':'kop','m':'njk','n':'bhjm',
                'o':'iklp','p':'ol','q':'wa','r':'edft','s':'awedxz','t':'rfgy','u':'yhji','v':'cfgb',
                'w':'qase','x':'zsdc','y':'tghu','z':'asx'
            }
            low = chars[i].lower()
            if low in neighbors:
                repl = random.choice(neighbors[low])
                chars[i] = repl.upper() if chars[i].isupper() else repl
        typo_word = ''.join(chars)
        type_fragment(service, doc_id, typo_word, start_time)
        time.sleep(random.uniform(0.1,0.4))
        pos = build_full_text(service, doc_id)[0].rfind(typo_word)
        if pos!=-1:
            safe_delete_range(service, doc_id, pos+1, pos+1+len(typo_word))

        # sometimes mimic an overzealous delete burst
        if random.random() < DELETE_BURST_PROB:
            delete_burst(service, doc_id)

    # now type the correct token in fragments
    for frag in fragment_word(token_to_type):
        type_fragment(service, doc_id, frag, start_time)

    # occasionally fix an earlier common mistake mid-stream
    if random.random() < MIDSTREAM_FIX_PROB:
        midstream_fix_prior_word(service, doc_id)

# ---------------- MAIN DRAFT ----------------
def human_draft(service, doc_id, text, start_time):
    source_pars = pick_paragraphs(text)
    tokens = re.findall(r"\w+|[^\w\s]|\s", text)
    i = 0
    prev_token = ''
    token_counter = 0

    # sometimes add a canned lead-in then tailor it
    paste_then_tailor(service, doc_id)

    while i < len(tokens):
        token = tokens[i]
        is_sentence_start = bool(re.match(r'\s*$', prev_token)) or prev_token in {'.','!','?','\n'}

        type_token_with_typos(service, doc_id, token, start_time, is_sentence_start)

        # quote-aware pause
        if '"' in token:
            time.sleep(random.uniform(*QUOTE_DELAY_RANGE))

        # paragraph boundary behavior: longer pause, maybe rearrange or add a heading
        if token == '\n' and i+1 < len(tokens) and tokens[i+1] == '\n':
            time.sleep(random.uniform(*PARA_START_RANGE))
            if random.random() < BACKTRACK_CHANCE:
                move_sentence(service, doc_id, source_pars)
            time.sleep(random.uniform(*PARA_DELAY_RANGE))
            if random.random() < 0.1:
                maybe_add_heading(service, doc_id)

        # after a sentence, sometimes rewrite it
        if token in {'.','!','?'} and random.random() < SENTENCE_REWRITE_PROB:
            rewrite_last_sentence(service, doc_id)

        # rare cleanup pass
        token_counter += 1
        if random.random() < CLEANUP_PASS_PROB or (token_counter % 300 == 0 and token_counter > 0):
            cleanup_spacing_caps(service, doc_id)

        prev_token = token
        i += 1

# ---------------- MAIN ----------------
def main():
    if not TEXT_TO_TYPE.strip():
        print("Paste your essay in TEXT_TO_TYPE in the script.")
        return
    print("Humanized Google Docs Typing Demo")
    service = authenticate()
    doc = service.documents().create(body={"title":"Humanized Draft"}).execute()
    doc_id = doc['documentId']
    print(f"Created doc: https://docs.google.com/document/d/{doc_id}/edit")

    start_time = time.time()
    human_draft(service, doc_id, TEXT_TO_TYPE, start_time)

if __name__=="__main__":
    main()