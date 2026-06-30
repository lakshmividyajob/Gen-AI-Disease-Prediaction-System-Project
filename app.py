"""
app.py  —  Disease Predictor AI  +  Generative AI (Claude)
5 Pages: Predict | AI Doctor Chat | ML Comparison | Data Insights | Model Deep Dive
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, warnings, os
from collections import Counter
warnings.filterwarnings("ignore")

import plotly.express       as px
import plotly.graph_objects as go
import anthropic

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Disease Predictor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = ["#6366f1","#06b6d4","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899"]

# ══════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');

  .stApp { background:#060d1a; color:#e2e8f0; font-family:'Space Grotesk',sans-serif; }

  [data-testid="stSidebar"] {
      background:linear-gradient(180deg,#0d1f3c,#060d1a);
      border-right:1px solid #1e3a5f;
  }

  /* Header */
  .header-box {
      background:linear-gradient(135deg,#1a3a6b 0%,#0e6b8a 50%,#0a4a6b 100%);
      border-radius:20px; padding:36px; text-align:center; margin-bottom:24px;
      border:1px solid #2a5a8a;
      box-shadow:0 0 60px rgba(14,107,138,0.4), inset 0 1px 0 rgba(255,255,255,0.1);
      position:relative; overflow:hidden;
  }
  .header-box::before {
      content:''; position:absolute; top:-50%; left:-50%;
      width:200%; height:200%;
      background:radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 60%);
      animation:pulse 4s ease-in-out infinite;
  }
  @keyframes pulse { 0%,100%{transform:scale(1);opacity:0.5} 50%{transform:scale(1.1);opacity:1} }
  .header-box h1 { color:white; font-size:2.4rem; margin:0; font-weight:700; position:relative; }
  .header-box p  { color:#7dd3fc; margin:8px 0 0; font-size:1.05rem; position:relative; }

  /* Cards */
  .card {
      background:linear-gradient(135deg,#0d1f3c,#0a1628);
      border:1px solid #1e3a5f; border-radius:14px;
      padding:20px; margin:10px 0;
      box-shadow:0 4px 24px rgba(0,0,0,0.4);
      transition: border-color 0.3s;
  }
  .card:hover { border-color:#3b82f6; }
  .card h3 { color:#38bdf8; margin:0 0 10px; font-size:1.1rem; }
  .card p  { color:#94a3b8; line-height:1.7; margin:0; }

  /* KPI */
  .kpi {
      background:linear-gradient(135deg,#0d1f3c,#0a1628);
      border:1px solid #1e3a5f; border-radius:12px;
      padding:18px; text-align:center;
      transition:transform 0.2s, border-color 0.2s;
  }
  .kpi:hover { transform:translateY(-3px); border-color:#6366f1; }
  .kpi .val { font-size:1.9rem; font-weight:700; color:#6366f1; }
  .kpi .lbl { font-size:0.78rem; color:#64748b; margin-top:5px; letter-spacing:0.05em; text-transform:uppercase; }

  /* Disease badge */
  .disease-badge {
      background:linear-gradient(135deg,#059669,#0891b2);
      border-radius:10px; padding:12px 26px; display:inline-block;
      font-size:1.4rem; font-weight:700; color:white; margin:8px 0;
      box-shadow:0 4px 20px rgba(5,150,105,0.4);
  }

  /* Pills */
  .pill {
      display:inline-block; background:rgba(29,78,216,0.3);
      border:1px solid #3b82f6; color:#93c5fd;
      border-radius:999px; padding:5px 14px; margin:4px; font-size:0.85rem;
  }

  /* Symptom tag */
  .sym-tag {
      display:inline-block; background:rgba(14,116,144,0.3);
      border:1px solid #0891b2; color:#67e8f9;
      border-radius:6px; padding:4px 12px; margin:3px; font-size:0.8rem;
  }

  .sev-low    { color:#4ade80; font-weight:600; }
  .sev-medium { color:#facc15; font-weight:600; }
  .sev-high   { color:#f87171; font-weight:600; }

  /* Warn */
  .warn-box {
      background:rgba(67,20,7,0.5); border:1px solid #ea580c;
      border-radius:10px; padding:14px 18px; color:#fed7aa; margin:12px 0;
  }

  /* Section title */
  .sec-title {
      background:linear-gradient(135deg,#1a3a6b,#0e6b8a);
      border-radius:8px; padding:10px 18px; margin:18px 0 10px;
      font-size:1rem; font-weight:700; color:white;
      border-left:4px solid #6366f1;
  }

  /* AI Chat bubble styles */
  .chat-user {
      background:linear-gradient(135deg,#1d4ed8,#1e40af);
      border-radius:16px 16px 4px 16px;
      padding:12px 18px; margin:8px 0; color:white;
      max-width:80%; margin-left:auto; font-size:0.95rem;
  }
  .chat-ai {
      background:linear-gradient(135deg,#0d1f3c,#0f2744);
      border:1px solid #1e3a5f; border-radius:16px 16px 16px 4px;
      padding:14px 18px; margin:8px 0; color:#e2e8f0;
      max-width:85%; font-size:0.95rem; line-height:1.7;
  }
  .chat-ai .ai-label {
      color:#38bdf8; font-weight:700; font-size:0.8rem;
      margin-bottom:6px; display:block;
  }

  /* Gen AI badge */
  .genai-badge {
      background:linear-gradient(135deg,#7c3aed,#4f46e5);
      border-radius:999px; padding:4px 14px; font-size:0.78rem;
      font-weight:700; color:white; display:inline-block; margin-left:8px;
      vertical-align:middle;
  }

  /* Button */
  .stButton>button {
      background:linear-gradient(135deg,#1d4ed8,#0891b2);
      color:white; border:none; border-radius:10px;
      padding:12px 30px; font-size:1rem; font-weight:600; width:100%;
      transition:opacity 0.2s, transform 0.2s;
  }
  .stButton>button:hover { opacity:0.88; transform:translateY(-1px); }

  .stMultiSelect [data-baseweb="tag"] { background:#1d4ed8 !important; color:white !important; }

  /* Report box */
  .report-box {
      background:linear-gradient(135deg,#0d1f3c,#071428);
      border:1px solid #2a5a8a; border-radius:14px;
      padding:24px; margin:16px 0; line-height:1.8;
      color:#cbd5e1; font-size:0.95rem;
  }

  div[data-testid="stMetricValue"] { color:#6366f1 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════
def pretty(s): return s.replace("_"," ").title()

def dark(fig):
    fig.update_layout(
        plot_bgcolor="#0d1f3c", paper_bgcolor="#060d1a",
        font_color="#e2e8f0",
        xaxis=dict(gridcolor="#1e3a5f"),
        yaxis=dict(gridcolor="#1e3a5f"),
        legend=dict(bgcolor="#0d1f3c", bordercolor="#1e3a5f", borderwidth=1),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════
# LOAD MODELS & DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    with open("model/trained_models.pkl",     "rb") as f: trained  = pickle.load(f)
    with open("model/symptoms_list.pkl",      "rb") as f: syms     = pickle.load(f)
    with open("model/conf_matrices.pkl",      "rb") as f: cms      = pickle.load(f)
    with open("model/feature_importance.pkl", "rb") as f: fi       = pickle.load(f)
    with open("model/test_data.pkl",          "rb") as f: td       = pickle.load(f)
    results = pd.read_csv("model/model_results.csv")
    return trained, syms, cms, fi, td, results

@st.cache_data
def load_csv():
    df      = pd.read_csv("data/dataset.csv")
    desc    = pd.read_csv("data/symptom_Description.csv")
    precaut = pd.read_csv("data/symptom_precaution.csv")
    sev     = pd.read_csv("data/Symptom-severity.csv")
    desc_dict, precaut_dict = {}, {}
    for _, r in desc.iterrows():    desc_dict[r["Disease"]]    = r["Description"]
    for _, r in precaut.iterrows():
        precaut_dict[r["Disease"]] = [r[c] for c in precaut.columns
                                      if c.startswith("Precaution") and pd.notna(r[c])]
    sev_dict = dict(zip(sev["Symptom"].str.strip(), sev["weight"]))
    return df, desc_dict, precaut_dict, sev_dict

try:
    trained, all_symptoms, cms, fi, (X_test, y_test), results_df = load_models()
    df, desc_dict, precaut_dict, sev_dict = load_csv()
    loaded = True
except Exception as e:
    loaded = False; load_err = str(e)

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🩺 Disease Predictor AI")
    st.markdown('<span class="genai-badge">✨ Gen AI</span>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("📌 Navigate", [
        "🔬 Predict Disease",
        "🤖 AI Doctor Chat",
        "📊 ML Model Comparison",
        "📈 Data Insights",
        "🔍 Model Deep Dive",
    ])

    st.markdown("---")
    if loaded:
        st.success("✅ 7 ML Models Ready")
        st.success("✨ Gen AI Ready")
        st.markdown(f"**🏆 Best Model:** {results_df.iloc[0]['Model']}")
        st.markdown(f"**🎯 Accuracy:** {results_df.iloc[0]['Accuracy']}%")
        st.markdown(f"**🦠 Diseases:** 41 | **🔬 Symptoms:** {len(all_symptoms)}")
    else:
        st.error("❌ Run train_models.py first!")
    st.markdown("---")

    # API Key input in sidebar
    st.markdown("### 🔑 Claude API Key")
    api_key = st.text_input(
        "Enter your Anthropic API key:",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com"
    )
    st.caption("Your key is never stored. [Get a free key →](https://console.anthropic.com)")
    st.markdown("---")
    st.caption("Python • Scikit-learn • Streamlit • Plotly • Claude AI")

if not loaded:
    st.error(f"⚠️ Please run `python train_models.py` first!\n\nError: {load_err}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════
# GEN AI HELPER — calls Claude API
# ══════════════════════════════════════════════════════════════════════
def call_claude(system_prompt, user_message, api_key):
    """Call Claude API and stream the response."""
    if not api_key or not api_key.startswith("sk-ant-"):
        st.warning("⚠️ Please enter a valid Claude API key in the sidebar to use AI features.")
        return None
    try:
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model      = "claude-opus-4-5",
            max_tokens = 1024,
            system     = system_prompt,
            messages   = [{"role": "user", "content": user_message}],
        ) as stream:
            return stream.get_final_text()
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT DISEASE
# ══════════════════════════════════════════════════════════════════════
if page == "🔬 Predict Disease":

    st.markdown("""
    <div class="header-box">
      <h1>🩺 AI Disease Predictor</h1>
      <p>Select symptoms → 7 ML models vote → Claude AI writes your health report</p>
    </div>""", unsafe_allow_html=True)

    sym_display = {pretty(s): s for s in sorted(all_symptoms)}

    col_l, col_r = st.columns([3,1])
    with col_l:
        selected_disp = st.multiselect(
            "Symptoms", list(sym_display.keys()),
            placeholder="Type a symptom e.g. Headache, Fever, Nausea...",
            label_visibility="collapsed",
        )
    with col_r:
        chosen_model = st.selectbox("ML Model:", list(trained.keys()))

    selected_raw = [sym_display[d] for d in selected_disp]

    if selected_raw:
        st.markdown("**Severity of your symptoms:**")
        for s in selected_raw:
            sev   = sev_dict.get(s, 1)
            cls   = "sev-high" if sev>=6 else ("sev-medium" if sev>=4 else "sev-low")
            label = "🔴 High"  if sev>=6 else ("🟡 Medium"  if sev>=4 else "🟢 Low")
            st.markdown(
                f'<span class="sym-tag">{pretty(s)}</span> <span class="{cls}">{label} (weight {sev})</span>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        predict_btn = st.button("🔬 Predict Disease", use_container_width=True)

    if predict_btn:
        if len(selected_raw) < 2:
            st.markdown('<div class="warn-box">⚠️ Please select at least 2 symptoms.</div>',
                        unsafe_allow_html=True)
        else:
            input_vec = {s:(1 if s in selected_raw else 0) for s in all_symptoms}
            input_df  = pd.DataFrame([input_vec])

            model      = trained[chosen_model]
            prediction = model.predict(input_df)[0]
            probs      = model.predict_proba(input_df)[0]
            classes    = model.classes_
            top3_idx   = probs.argsort()[::-1][:3]

            st.markdown("---")
            st.markdown("## 📊 Prediction Results")

            col_a, col_b = st.columns([3,1])
            with col_a:
                st.markdown("### 🏥 Most Likely Disease")
                st.markdown(f'<div class="disease-badge">✅ {prediction}</div>',
                            unsafe_allow_html=True)
                conf = probs[list(classes).index(prediction)] * 100
                st.progress(min(int(conf),100))
                st.markdown(f"**Confidence: {conf:.1f}%** — *{chosen_model}*")
            with col_b:
                st.markdown("### 📈 Top 3")
                for i in top3_idx:
                    st.markdown(f"**{classes[i]}**  \n`{probs[i]*100:.1f}%`")

            # All 7 models vote
            st.markdown('<div class="sec-title">🗳️ All 7 Models Vote</div>', unsafe_allow_html=True)
            votes, model_preds = {}, {}
            for mname, m in trained.items():
                pred = m.predict(input_df)[0]
                model_preds[mname] = pred
                votes[pred] = votes.get(pred, 0) + 1

            cols = st.columns(len(trained))
            for i,(mname,pred) in enumerate(model_preds.items()):
                cols[i].markdown(f"**{mname}**\n\n{'✅' if pred==prediction else '🟡'} `{pred}`")

            vote_df = pd.DataFrame(list(votes.items()), columns=["Disease","Votes"]).sort_values("Votes", ascending=False)
            fig_v = px.bar(vote_df, x="Disease", y="Votes",
                           color="Votes", color_continuous_scale="Blues", text="Votes",
                           title="Model Votes per Disease")
            fig_v.update_traces(textposition="outside")
            dark(fig_v); fig_v.update_layout(coloraxis_showscale=False, title_font_color="#38bdf8")
            st.plotly_chart(fig_v, use_container_width=True)

            # Probability chart
            st.markdown('<div class="sec-title">🎯 Top 10 Disease Probabilities</div>', unsafe_allow_html=True)
            prob_df = pd.DataFrame({"Disease":classes,"Probability":probs*100}).sort_values("Probability",ascending=False).head(10)
            fig_p = px.bar(prob_df, x="Probability", y="Disease", orientation="h",
                           color="Probability", color_continuous_scale="Teal",
                           text=prob_df["Probability"].round(1).astype(str)+"%")
            fig_p.update_traces(textposition="outside")
            dark(fig_p); fig_p.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("---")

            # Description & Precautions
            if prediction in desc_dict:
                st.markdown(f"""<div class="card"><h3>📖 About {prediction}</h3>
                <p>{desc_dict[prediction]}</p></div>""", unsafe_allow_html=True)

            if prediction in precaut_dict:
                pills = " ".join(f'<span class="pill">✔ {p.strip().title()}</span>'
                                 for p in precaut_dict[prediction])
                st.markdown(f"""<div class="card"><h3>💊 Precautions &amp; Cure</h3>
                <div style="margin-top:8px;">{pills}</div></div>""", unsafe_allow_html=True)

            # Severity radar
            st.markdown('<div class="sec-title">🕸️ Symptom Severity Radar</div>', unsafe_allow_html=True)
            sev_vals  = [sev_dict.get(s,1) for s in selected_raw]
            sev_names = [pretty(s) for s in selected_raw]
            if len(sev_names) >= 3:
                fig_r = go.Figure(go.Scatterpolar(
                    r=sev_vals+[sev_vals[0]], theta=sev_names+[sev_names[0]],
                    fill="toself", line_color="#6366f1", fillcolor="rgba(99,102,241,0.2)",
                ))
                fig_r.update_layout(
                    polar=dict(bgcolor="#0d1f3c",
                               radialaxis=dict(visible=True,range=[0,7],color="#94a3b8"),
                               angularaxis=dict(color="#94a3b8")),
                    paper_bgcolor="#060d1a", font_color="#e2e8f0",
                )
                st.plotly_chart(fig_r, use_container_width=True)

            # ── ✨ GEN AI HEALTH REPORT ────────────────────────────────────────
            st.markdown('<div class="sec-title">✨ AI Health Report <span class="genai-badge">Claude AI</span></div>',
                        unsafe_allow_html=True)

            if api_key:
                if st.button("🧠 Generate My AI Health Report", use_container_width=True):
                    syms_text = ", ".join([pretty(s) for s in selected_raw])
                    precautions_text = ", ".join(precaut_dict.get(prediction, []))

                    system = """You are a helpful, friendly medical AI assistant. 
                    You provide clear, easy-to-understand health information. 
                    Always remind users to consult a real doctor. 
                    Format your response with clear sections using emojis."""

                    prompt = f"""
                    A patient has the following symptoms: {syms_text}
                    The ML model predicted the disease as: {prediction}
                    Known precautions: {precautions_text}

                    Please write a comprehensive but easy-to-understand health report covering:
                    1. 🦠 What this disease is (in simple words)
                    2. ⚠️ Why these symptoms match
                    3. 🥗 Foods to eat and avoid
                    4. 💊 General treatment approach
                    5. 🏥 When to see a doctor urgently
                    6. 💪 Recovery tips and lifestyle advice

                    Keep it warm, clear and helpful. Add a disclaimer at the end.
                    """

                    with st.spinner("🧠 Claude AI is writing your health report..."):
                        report = call_claude(system, prompt, api_key)

                    if report:
                        st.markdown(f'<div class="report-box">{report}</div>',
                                    unsafe_allow_html=True)
            else:
                st.info("💡 Enter your Claude API key in the sidebar to get a personalized AI health report!")

            st.markdown("""<div class="warn-box">
            ⚠️ <strong>Disclaimer:</strong> For educational purposes only. Always consult a real doctor.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — AI DOCTOR CHAT
# ══════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Doctor Chat":

    st.markdown("""
    <div class="header-box">
      <h1>🤖 AI Doctor Chat</h1>
      <p>Ask any health question — powered by Claude AI <span class="genai-badge">Gen AI</span></p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <h3>💬 What can I ask?</h3>
      <p>
      • "What foods should I eat if I have diabetes?" <br>
      • "Explain the difference between flu and cold symptoms" <br>
      • "What are early signs of hypertension?" <br>
      • "How can I naturally manage migraines?" <br>
      • "What vitamins help with fatigue?"
      </p>
    </div>""", unsafe_allow_html=True)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-ai">
              <span class="ai-label">🤖 AI Doctor</span>
              {msg["content"]}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Quick question buttons
    st.markdown("**⚡ Quick Questions:**")
    q_cols = st.columns(3)
    quick_qs = [
        "What foods help boost immunity?",
        "How to manage stress naturally?",
        "Signs I should see a doctor urgently?",
        "Best diet for diabetes?",
        "How to improve sleep quality?",
        "Natural remedies for headaches?",
    ]
    for i, q in enumerate(quick_qs):
        if q_cols[i % 3].button(q, key=f"quick_{i}"):
            st.session_state.pending_question = q

    # Text input
    user_input = st.chat_input("Ask any health question...")

    # Handle question (either typed or quick button)
    question = user_input
    if hasattr(st.session_state, "pending_question"):
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question:
        st.session_state.chat_history.append({"role":"user","content":question})

        system = """You are a friendly, knowledgeable AI health assistant. 
        Provide helpful, accurate, easy-to-understand health information.
        Use emojis to make responses more readable. Keep responses concise but complete.
        Always remind users that this is general information and they should consult a doctor for personal medical advice.
        Never diagnose. Always be warm and supportive."""

        with st.spinner("🤖 AI Doctor is thinking..."):
            response = call_claude(system, question, api_key)

        if response:
            st.session_state.chat_history.append({"role":"assistant","content":response})
            st.rerun()

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    if not api_key:
        st.info("💡 Enter your Claude API key in the sidebar to start chatting with the AI Doctor!")


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — ML MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════
elif page == "📊 ML Model Comparison":

    st.markdown("# 📊 ML Model Comparison")
    st.markdown("*All 7 models trained and tested on the real Kaggle dataset*")

    best = results_df.iloc[0]
    st.markdown(f"""<div class="card">
      <h3>🏆 Best Model: {best['Model']}</h3>
      <div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:8px;">
        <span style="color:#64748b;">Accuracy:  <strong style="color:#10b981;">{best['Accuracy']}%</strong></span>
        <span style="color:#64748b;">Precision: <strong style="color:#10b981;">{best['Precision']}%</strong></span>
        <span style="color:#64748b;">Recall:    <strong style="color:#10b981;">{best['Recall']}%</strong></span>
        <span style="color:#64748b;">F1 Score:  <strong style="color:#10b981;">{best['F1 Score']}%</strong></span>
        <span style="color:#64748b;">CV Score:  <strong style="color:#10b981;">{best['CV Score']}%</strong></span>
      </div></div>""", unsafe_allow_html=True)

    # Scorecard
    st.markdown('<div class="sec-title">📋 Full Scorecard — All 7 Models</div>', unsafe_allow_html=True)
    st.dataframe(
        results_df.style.background_gradient(
            subset=["Accuracy","Precision","Recall","F1 Score","CV Score"], cmap="Blues"
        ), use_container_width=True, hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">🎯 Accuracy per Model</div>', unsafe_allow_html=True)
        fig = px.bar(results_df.sort_values("Accuracy"), x="Accuracy", y="Model",
                     orientation="h", color="Accuracy",
                     color_continuous_scale="Blues", text="Accuracy", range_x=[0,110])
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        dark(fig); fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">🕸️ Multi-Metric Radar</div>', unsafe_allow_html=True)
        metrics = ["Accuracy","Precision","Recall","F1 Score","CV Score"]
        fig2 = go.Figure()
        for i, (_, row) in enumerate(results_df.iterrows()):
            vals = [row[m] for m in metrics]
            fig2.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=metrics+[metrics[0]],
                fill="toself", name=row["Model"],
                line_color=COLORS[i%len(COLORS)], opacity=0.7,
            ))
        fig2.update_layout(
            polar=dict(bgcolor="#0d1f3c",
                       radialaxis=dict(visible=True,range=[0,105],color="#94a3b8"),
                       angularaxis=dict(color="#94a3b8")),
            paper_bgcolor="#060d1a", font_color="#e2e8f0",
            legend=dict(bgcolor="#0d1f3c",bordercolor="#1e3a5f",borderwidth=1),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Grouped metrics bar
    st.markdown('<div class="sec-title">📊 All Metrics Side-by-Side</div>', unsafe_allow_html=True)
    melted = results_df.melt(id_vars="Model",
                              value_vars=["Accuracy","Precision","Recall","F1 Score","CV Score"],
                              var_name="Metric", value_name="Score")
    fig3 = px.bar(melted, x="Model", y="Score", color="Metric",
                  barmode="group", color_discrete_sequence=COLORS,
                  text=melted["Score"].astype(str)+"%")
    fig3.update_traces(textposition="outside", textfont_size=9)
    dark(fig3); fig3.update_layout(xaxis_tickangle=-20, yaxis_range=[0,115])
    st.plotly_chart(fig3, use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="sec-title">🔢 Confusion Matrix</div>', unsafe_allow_html=True)
    sel = st.selectbox("Select model:", list(cms.keys()))
    cm  = cms[sel]
    cl  = list(trained[sel].classes_)
    n   = min(20, len(cl))
    fig4 = px.imshow(cm[:n,:n], text_auto=True,
                     x=cl[:n], y=cl[:n],
                     color_continuous_scale="Blues",
                     labels=dict(x="Predicted",y="Actual"),
                     title=f"Confusion Matrix — {sel}")
    fig4.update_layout(paper_bgcolor="#060d1a", font_color="#e2e8f0",
                       title_font_color="#38bdf8", xaxis_tickangle=-40, height=600)
    st.plotly_chart(fig4, use_container_width=True)

    # Feature importance
    st.markdown('<div class="sec-title">⭐ Top 20 Most Important Symptoms (Random Forest)</div>',
                unsafe_allow_html=True)
    fig5 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Teal", text=fi["Importance"].round(4))
    fig5.update_traces(textposition="outside")
    dark(fig5); fig5.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

    # ✨ AI model explanation
    st.markdown('<div class="sec-title">✨ AI Explains the Results <span class="genai-badge">Claude AI</span></div>',
                unsafe_allow_html=True)
    if api_key:
        if st.button("🧠 Ask AI to Explain These ML Results", use_container_width=True):
            results_text = results_df.to_string(index=False)
            system = "You are an expert data scientist who explains ML results to beginners in a friendly, clear way using analogies and simple language."
            prompt = f"""
            Here are the results of 7 ML models trained on a disease prediction dataset:
            {results_text}

            Please explain:
            1. 🏆 Why some models scored higher than others
            2. 📊 What Accuracy, Precision, Recall and F1 Score mean in simple words
            3. 🌲 Why Random Forest is often a top performer
            4. ⚠️ What the low AdaBoost score might mean
            5. 💡 Which model would you recommend for this medical use case and why
            """
            with st.spinner("🧠 AI is analyzing the results..."):
                explanation = call_claude(system, prompt, api_key)
            if explanation:
                st.markdown(f'<div class="report-box">{explanation}</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Add your API key in the sidebar to get an AI explanation of these results!")


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════
elif page == "📈 Data Insights":

    st.markdown("# 📈 Dataset Insights")
    st.markdown("*Exploring the Kaggle Disease-Symptom dataset visually*")

    symptom_cols  = [c for c in df.columns if c.startswith("Symptom")]
    all_syms_flat = [s.strip() for s in df[symptom_cols].values.flatten()
                     if pd.notna(s) and str(s).strip()]

    k1,k2,k3,k4 = st.columns(4)
    for col,val,lbl in zip([k1,k2,k3,k4],
                            [df["Disease"].nunique(), len(df), len(set(all_syms_flat)), len(symptom_cols)],
                            ["Unique Diseases","Total Records","Unique Symptoms","Symptom Columns"]):
        col.markdown(f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("---")
    col1,col2 = st.columns(2)

    with col1:
        st.markdown('<div class="sec-title">🦠 Records per Disease</div>', unsafe_allow_html=True)
        dc = df["Disease"].value_counts().reset_index()
        dc.columns = ["Disease","Count"]
        fig = px.bar(dc, x="Count", y="Disease", orientation="h",
                     color="Count", color_continuous_scale="Viridis", height=800)
        dark(fig); fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">🔬 Top 20 Most Common Symptoms</div>', unsafe_allow_html=True)
        sym_df = pd.DataFrame(Counter(all_syms_flat).most_common(20), columns=["Symptom","Count"])
        sym_df["Symptom"] = sym_df["Symptom"].apply(pretty)
        fig2 = px.bar(sym_df, x="Count", y="Symptom", orientation="h",
                      color="Count", color_continuous_scale="Plasma")
        dark(fig2); fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Severity histogram
    st.markdown('<div class="sec-title">⚖️ Severity Weight Distribution</div>', unsafe_allow_html=True)
    sev_df = pd.read_csv("data/Symptom-severity.csv")
    fig3 = px.histogram(sev_df, x="weight", nbins=7,
                        color_discrete_sequence=[COLORS[0]],
                        labels={"weight":"Severity Weight"},
                        title="Symptoms per Severity Level")
    dark(fig3); st.plotly_chart(fig3, use_container_width=True)

    # Avg symptom count per disease
    st.markdown('<div class="sec-title">📐 Average Symptoms per Disease</div>', unsafe_allow_html=True)
    df["sym_count"] = df[symptom_cols].notna().sum(axis=1)
    sc = df.groupby("Disease")["sym_count"].mean().reset_index().sort_values("sym_count",ascending=False)
    fig4 = px.bar(sc, x="Disease", y="sym_count",
                  color="sym_count", color_continuous_scale="Magma",
                  labels={"sym_count":"Avg Symptom Count"})
    dark(fig4); fig4.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    # Heatmap
    st.markdown('<div class="sec-title">🔥 Disease × Symptom Heatmap (Top 10 Diseases)</div>',
                unsafe_allow_html=True)
    top10 = df["Disease"].value_counts().head(10).index.tolist()
    top15s = [s for s,_ in Counter(all_syms_flat).most_common(15)]
    rows = []
    for disease in top10:
        sub = df[df["Disease"]==disease]
        row = {"Disease":disease}
        for sym in top15s:
            row[pretty(sym)] = sum(sub[c].str.strip().eq(sym).sum()
                                   for c in symptom_cols if c in sub)
        rows.append(row)
    heat_df = pd.DataFrame(rows).set_index("Disease")
    fig5 = px.imshow(heat_df, color_continuous_scale="Blues",
                     text_auto=True, aspect="auto")
    fig5.update_layout(paper_bgcolor="#060d1a", font_color="#e2e8f0", xaxis_tickangle=-30)
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL DEEP DIVE
# ══════════════════════════════════════════════════════════════════════
elif page == "🔍 Model Deep Dive":

    st.markdown("# 🔍 Model Deep Dive")
    st.markdown("*Inspect any individual model in full detail*")

    sel = st.selectbox("Choose a model:", list(trained.keys()))
    row = results_df[results_df["Model"]==sel].iloc[0]

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,metric in zip([c1,c2,c3,c4,c5],
                           ["Accuracy","Precision","Recall","F1 Score","CV Score"]):
        col.markdown(f'<div class="kpi"><div class="val">{row[metric]}%</div>'
                     f'<div class="lbl">{metric}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    model  = trained[sel]
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})

    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="sec-title">🦠 Correct Predictions per Disease</div>', unsafe_allow_html=True)
        correct = pred_df[pred_df["Actual"]==pred_df["Predicted"]]["Actual"].value_counts().reset_index()
        correct.columns = ["Disease","Correct"]
        fig = px.bar(correct, x="Disease", y="Correct",
                     color="Correct", color_continuous_scale="Greens")
        dark(fig); fig.update_layout(xaxis_tickangle=-45, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="sec-title">🎯 Metric Gauge</div>', unsafe_allow_html=True)
        metrics = ["Accuracy","Precision","Recall","F1 Score","CV Score"]
        vals    = [row[m] for m in metrics]
        fig2 = go.Figure(go.Bar(
            x=vals, y=metrics, orientation="h",
            marker=dict(color=vals, colorscale="RdYlGn", cmin=0, cmax=100),
            text=[f"{v}%" for v in vals], textposition="outside",
        ))
        fig2.update_layout(paper_bgcolor="#060d1a", plot_bgcolor="#0d1f3c",
                           font_color="#e2e8f0", xaxis_range=[0,115],
                           xaxis=dict(gridcolor="#1e3a5f"),
                           yaxis=dict(gridcolor="#1e3a5f"), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Confusion matrix
    st.markdown(f'<div class="sec-title">🔢 Confusion Matrix — {sel}</div>', unsafe_allow_html=True)
    cm  = cms[sel]; cl = list(model.classes_); n = min(20,len(cl))
    fig3 = px.imshow(cm[:n,:n], text_auto=True, x=cl[:n], y=cl[:n],
                     color_continuous_scale="Blues",
                     labels=dict(x="Predicted",y="Actual"))
    fig3.update_layout(paper_bgcolor="#060d1a", font_color="#e2e8f0",
                       xaxis_tickangle=-40, height=600)
    st.plotly_chart(fig3, use_container_width=True)

    # Sample predictions
    st.markdown('<div class="sec-title">🧾 Sample Predictions vs Actual</div>', unsafe_allow_html=True)
    sample = pred_df.sample(min(30,len(pred_df)), random_state=42).reset_index(drop=True)
    sample["✅ Correct"] = sample["Actual"] == sample["Predicted"]
    st.dataframe(
        sample.style.apply(
            lambda col: ["background-color:#052e16" if v else "background-color:#450a0a"
                         for v in col] if col.name=="✅ Correct" else [""]*len(col), axis=0
        ), use_container_width=True, height=400,
    )

    # ✨ AI explains this model
    st.markdown('<div class="sec-title">✨ AI Explains This Model <span class="genai-badge">Claude AI</span></div>',
                unsafe_allow_html=True)
    if api_key:
        if st.button(f"🧠 Ask AI to Explain {sel}", use_container_width=True):
            system = "You are a friendly data science teacher explaining ML models to beginners."
            prompt = f"""
            Explain the {sel} machine learning model to a complete beginner.
            Its performance metrics are:
            - Accuracy:  {row['Accuracy']}%
            - Precision: {row['Precision']}%
            - Recall:    {row['Recall']}%
            - F1 Score:  {row['F1 Score']}%
            - CV Score:  {row['CV Score']}%

            Cover:
            1. 🤔 How does {sel} work? (use a simple analogy)
            2. ✅ Strengths of this model
            3. ❌ Weaknesses of this model
            4. 🏥 Is it suitable for disease prediction?
            5. 💡 Tips to improve its performance
            """
            with st.spinner(f"🧠 AI is analyzing {sel}..."):
                explanation = call_claude(system, prompt, api_key)
            if explanation:
                st.markdown(f'<div class="report-box">{explanation}</div>', unsafe_allow_html=True)
    else:
        st.info("💡 Add your API key in the sidebar to get an AI explanation!")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.8rem;padding:8px 0;">
  🩺 Disease Predictor AI &nbsp;•&nbsp; 7 ML Models &nbsp;•&nbsp; Claude Gen AI
  &nbsp;•&nbsp; Kaggle Dataset &nbsp;•&nbsp; Educational Use Only
</div>
""", unsafe_allow_html=True)
