"""
Human vs AI Writing Pattern Analyzer — Streamlit frontend
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

from features import extract_features, repeated_phrases, text_stats

# ---------------------------------------------------------------- config
MIN_WORDS, MSAMPLIN_SENTENCES, MAX_WORDS = 100, 3, 2000
SHORT_TEXT_WORDS = 150            # between MIN and this -> "less reliable" note

HUMAN, AI, INK, MUTED, LINE = "#2F7D6D", "#9C3D6B", "#1D2433", "#6B7385", "#DDE1E8"

FRIENDLY = {
    "avg_sent_len": "Average sentence length", "std_sent_len": "Sentence length variation",
    "min_sent_len": "Shortest sentence", "max_sent_len": "Longest sentence",
    "ttr": "Vocabulary diversity", "avg_word_len": "Average word length",
    "long_word_pct": "Long words (7+ letters)", "bigram_repeat": "Repeated 2-word phrases",
    "trigram_repeat": "Repeated 3-word phrases", "burstiness": "Burstiness",
    "formality": "Formality", "contraction_rate": "Contractions",
    "first_person_rate": "First-person words", "passive_rate": "Passive voice",
    "n_paragraphs": "Number of paragraphs", "avg_para_len": "Average paragraph length",
    "para_len_cv": "Paragraph length variation", "flesch_ease": "Reading ease",
    "fk_grade": "Grade level", "noun_pct": "Nouns", "verb_pct": "Verbs",
    "adj_pct": "Adjectives", "adv_pct": "Adverbs", "pron_pct": "Pronouns",
    "prep_pct": "Prepositions", "comma_rate": "Commas per sentence",
    "question_rate": "Questions per sentence",
}

CHARACTERISTICS = {
    "Sentence variation": "std_sent_len",
    "Vocabulary diversity": "ttr",
    "Repetition": "trigram_repeat",
    "Burstiness": "burstiness",
    "Formality": "formality",
    "Paragraph structure": "para_len_cv",
    "Readability (ease)": "flesch_ease",
    "Long-word usage": "long_word_pct",
}

E_HUMAN = """I honestly don't think phones should be banned in school. Sure, some kids play games in class. But that's not the phone's fault, it's theirs.

Last year my friend forgot her inhaler and she had to call her mom from the nurse's office, which took forever because the line was busy. If she'd had her phone it would've taken two minutes. Teachers can't watch everyone all the time, and they know that too.

I get why parents worry. Still, taking phones away from everybody because a few people misuse them just seems unfair to me. Why not teach us how to use them properly instead? We're going to have them for the rest of our lives anyway."""

SAMPLE_AI = """Mobile phones have become an integral part of modern education, offering both significant opportunities and notable challenges for students and teachers alike.

On one hand, smartphones provide immediate access to educational resources, enabling students to research topics, collaborate on projects, and develop essential digital literacy skills. Furthermore, they can serve as valuable tools for communication in emergency situations, ensuring student safety and peace of mind for parents.

On the other hand, excessive phone usage can lead to distractions that negatively impact academic performance. Therefore, rather than implementing a complete ban, schools should establish clear guidelines that promote responsible usage. By striking a balance between accessibility and discipline, educational institutions can harness the benefits of technology while minimizing its potential drawbacks."""

st.set_page_config(page_title="Writing Pattern Analyzer", page_icon="✍️", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea {{ font-family: 'Public Sans', sans-serif; }}
h1, h2, h3 {{ font-family: 'Newsreader', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.4rem !important; font-weight: 600 !important; padding-bottom: 0.2rem; }}
h3 {{ font-size: 1.35rem !important; }}
.block-container {{ max-width: 1080px; padding-top: 2.2rem; }}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stToolbar"],
#MainMenu, footer {{ display: none !important; }}
.verdict {{ border-left: 6px solid var(--c); padding: 0.5rem 0 0.5rem 1.1rem; margin: 0.6rem 0 0.8rem; }}
.verdict .label {{ font-family: 'Newsreader', Georgia, serif; font-size: 2.2rem; font-weight: 600; color: var(--c); line-height: 1.1; }}
.verdict .sub {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.2rem; }}
.scale {{ position: relative; height: 10px; border-radius: 5px; margin: 1.4rem 0 0.4rem;
          background: linear-gradient(90deg, {HUMAN} 0%, {LINE} 50%, {AI} 100%); }}
.scale .pin {{ position: absolute; top: -7px; width: 4px; height: 24px; background: {INK}; border-radius: 2px; transform: translateX(-2px); }}
.scale-labels {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: 0.85rem; }}
.reason {{ padding: 0.5rem 0; border-bottom: 1px solid {LINE}; }}
.reason b {{ color: {INK}; }}
.tag {{ font-size: 0.8rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 999px; color: white; white-space: nowrap; }}
.foot {{ color: {MUTED}; font-size: 0.85rem; margin-top: 2rem; border-top: 1px solid {LINE}; padding-top: 0.8rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
@st.cache_resource
def load_artifacts():
    art = joblib.load("writing_analyzer_meta.joblib")
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    art["model"] = model
    art["importance"] = pd.Series(model.feature_importances_,
                                  index=art["features"]).sort_values(ascending=False)
    return art


try:
    ART = load_artifacts()
except Exception as e:
    st.error("Could not load the model. Make sure `xgb_model.json` and `writing_analyzer_meta.joblib` "
             f"are in the same folder as app.py.\n\nDetails: {e}")
    st.stop()


# ---------------------------------------------------------------- logic (cached so reruns are instant)
@st.cache_data(show_spinner=False, max_entries=100)
def cached_stats(text):
    return text_stats(text)


@st.cache_data(show_spinner=False, max_entries=50)
def analyze(text):
    feats = extract_features(text)
    X = pd.DataFrame([feats])[ART["features"]]
    p_ai = float(ART["model"].predict_proba(X)[0, 1])
    return {"text": text, "feats": feats, "p_ai": p_ai,
            "label": "AI-like" if p_ai >= 0.5 else "Human-like",
            "stats": cached_stats(text)}


def level(feat, value):
    t = ART["thresholds"].loc[feat]
    return "Low" if value < t["low"] else "High" if value > t["high"] else "Medium"


def closer_to(feat, value):
    m = ART["class_means"].loc[feat]
    return "AI" if abs(value - m["AI"]) < abs(value - m["Human"]) else "Human"


def validate(text):
    """Return (error, warning) for the input text."""
    s = cached_stats(text)
    if s["n_words"] < MIN_WORDS:
        return f"Add more text: {s['n_words']} words so far, at least {MIN_WORDS} are needed.", None
    if s["n_sentences"] < MIN_SENTENCES:
        return f"Use at least {MIN_SENTENCES} full sentences so sentence patterns can be measured.", None
    if s["n_words"] > MAX_WORDS:
        return f"Shorten the text to {MAX_WORDS} words or fewer (currently {s['n_words']}).", None
    if s["n_words"] < SHORT_TEXT_WORDS:
        return None, f"Short text ({s['n_words']} words) gives less reliable results. 150+ words works best."
    return None, None


# ---------------------------------------------------------------- UI pieces
def counter(text):
    s = cached_stats(text) if text.strip() else {"n_words": 0, "n_sentences": 0}
    st.progress(min(s["n_words"] / MIN_WORDS, 1.0))
    need = MIN_WORDS - s["n_words"]
    note = "ready to analyze" if need <= 0 else f"{need} more words needed"
    st.caption(f"{s['n_words']} words · {s['n_sentences']} sentences · {note}")


def verdict_block(r, title=None):
    color = AI if r["label"] == "AI-like" else HUMAN
    conf = max(r["p_ai"], 1 - r["p_ai"])
    head = f"<div class='sub'>{title}</div>" if title else ""
    st.markdown(f"""
    <div class='verdict' style='--c:{color}'>
      {head}<div class='label'>{r['label']} writing</div>
      <div class='sub'>Model confidence {conf:.0%} · {r['stats']['n_words']} words analyzed</div>
      <div class='scale'><div class='pin' style='left:{r['p_ai']*100:.1f}%'></div></div>
      <div class='scale-labels'><span>More human-like</span><span>More AI-like</span></div>
    </div>""", unsafe_allow_html=True)


def characteristics_table(feats):
    return pd.DataFrame([{"Characteristic": n, "Value": round(feats[f], 2),
                          "Level": level(f, feats[f]), "Closer to": closer_to(f, feats[f])}
                         for n, f in CHARACTERISTICS.items()])


def reasons(r, n=6):
    top = ART["importance"].index[:n]
    pred = "AI" if r["label"] == "AI-like" else "Human"
    items = [(f, level(f, r["feats"][f]), closer_to(f, r["feats"][f])) for f in top]
    items.sort(key=lambda x: x[2] != pred)          # supporting reasons first
    html = ""
    for f, lv, side in items:
        c = AI if side == "AI" else HUMAN
        html += (f"<div class='reason'><b>{FRIENDLY.get(f, f)}</b>: {lv.lower()} "
                 f"<span class='tag' style='background:{c}'>closer to {side}</span></div>")
    st.markdown(html, unsafe_allow_html=True)


def fig_style(fig, h=300):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Public Sans", color=INK),
                      legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor=LINE); fig.update_yaxes(gridcolor=LINE)
    return fig


def rhythm_chart(results):
    fig = go.Figure()
    for name, r, color in results:
        lens = r["stats"]["sentence_lengths"]
        fig.add_bar(x=list(range(1, len(lens) + 1)), y=lens, name=name, marker_color=color, opacity=0.85)
    fig.update_layout(barmode="group", xaxis_title="Sentence number", yaxis_title="Words")
    return fig_style(fig)


def show_results(r):
    if st.session_state.get("warn"):
        st.warning(st.session_state["warn"])
    verdict_block(r)

    t_why, t_charts, t_all = st.tabs(["Why this result", "Charts", "All features"])

    with t_why:
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Writing characteristics")
            st.dataframe(characteristics_table(r["feats"]), hide_index=True, width="stretch")
        with right:
            st.subheader("Main reasons")
            st.caption("Most important features, compared with the average human and AI text.")
            reasons(r)

    with t_charts:
        st.subheader("Sentence rhythm")
        st.caption("Each bar is one sentence. Even bars = low burstiness, uneven bars = high burstiness.")
        st.plotly_chart(rhythm_chart([("Sentence length", r, AI if r["label"] == "AI-like" else HUMAN)]),
                        width="stretch")
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.subheader("Parts of speech")
            pos = {k: r["feats"][f] for k, f in [("Nouns", "noun_pct"), ("Verbs", "verb_pct"),
                   ("Adjectives", "adj_pct"), ("Adverbs", "adv_pct"), ("Pronouns", "pron_pct"),
                   ("Prepositions", "prep_pct")]}
            fig = go.Figure(go.Bar(x=list(pos.values()), y=list(pos.keys()), orientation="h", marker_color=INK))
            fig.update_layout(xaxis_title="% of words")
            st.plotly_chart(fig_style(fig), width="stretch")
        with g2:
            st.subheader("Repeated phrases")
            phrases = repeated_phrases(r["text"], 2) + repeated_phrases(r["text"], 3)
            if phrases:
                st.dataframe(pd.DataFrame(sorted(phrases, key=lambda x: -x[1])[:10], columns=["Phrase", "Times"]),
                             hide_index=True, width="stretch")
            else:
                st.write("No phrase is repeated in this text.")

    with t_all:
        st.dataframe(pd.DataFrame([{"Feature": FRIENDLY.get(k, k), "Value": round(v, 3), "Level": level(k, v)}
                                   for k, v in r["feats"].items()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- page
st.title("Writing Pattern Analyzer")

tab_one, tab_two = st.tabs(["Analyze a text", "Compare two texts"])

# ---- Tab 1
with tab_one:
    def set_sample(t):
        st.session_state["text_input"] = t
        st.session_state.pop("result", None)

    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("Human sample", on_click=set_sample, args=(SAMPLE_HUMAN,))
    c2.button("AI sample", on_click=set_sample, args=(SAMPLE_AI,))

    text = st.text_area("Your text", key="text_input", height=220, label_visibility="collapsed",
                        placeholder=f"Paste at least {MIN_WORDS} words here…")
    with st.expander("Or upload a .txt file"):
        uploaded = st.file_uploader("Text file", type=["txt"], label_visibility="collapsed")
    if uploaded is not None:
        text = uploaded.read().decode("utf-8", errors="ignore")
        st.caption(f"Using uploaded file: {uploaded.name}")
    counter(text)

    if st.button("Analyze text", type="primary"):
        err, warn = validate(text)
        if err:
            st.error(err)
            st.session_state.pop("result", None)
        else:
            with st.spinner("Measuring writing patterns…"):
                st.session_state["result"] = analyze(text)
            st.session_state["warn"] = warn

    if st.session_state.get("result"):
        show_results(st.session_state["result"])

# ---- Tab 2
with tab_two:
    a, b = st.columns(2, gap="large")
    with a:
        text_a = st.text_area("Text A", height=200, key="text_a", placeholder=f"Text A: at least {MIN_WORDS} words")
        counter(text_a)
    with b:
        text_b = st.text_area("Text B", height=200, key="text_b", placeholder=f"Text B: at least {MIN_WORDS} words")
        counter(text_b)

    if st.button("Compare texts", type="primary"):
        errors = [f"Text {n}: {e}" for n, t in [("A", text_a), ("B", text_b)] if (e := validate(t)[0])]
        if errors:
            for e in errors:
                st.error(e)
            st.session_state.pop("compare", None)
        else:
            with st.spinner("Measuring both texts…"):
                st.session_state["compare"] = (analyze(text_a), analyze(text_b))

    if st.session_state.get("compare"):
        ra, rb = st.session_state["compare"]
        v1, v2 = st.columns(2, gap="large")
        with v1: verdict_block(ra, "Text A")
        with v2: verdict_block(rb, "Text B")

        ta, tb = characteristics_table(ra["feats"]), characteristics_table(rb["feats"])
        st.subheader("Side by side")
        st.dataframe(pd.DataFrame({"Characteristic": ta["Characteristic"],
                                   "Text A": ta["Level"], "Text B": tb["Level"],
                                   "A closer to": ta["Closer to"], "B closer to": tb["Closer to"]}),
                     hide_index=True, width="stretch")
        st.subheader("Sentence rhythm")
        st.plotly_chart(rhythm_chart([("Text A", ra, "#2D4B8E"), ("Text B", rb, "#C98A2E")]), width="stretch")

st.markdown("<div class='foot'>Results are predictions based on writing-style patterns in the training data, "
            "not proof of who wrote a text.</div>", unsafe_allow_html=True)