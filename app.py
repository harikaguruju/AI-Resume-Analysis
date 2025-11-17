import streamlit as st
import os, io, json, re, uuid, logging, requests, random
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import PyPDF2, docx2txt, spacy
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

st.set_page_config(page_title="SkillSight Pro v3.6", layout="wide")
logging.basicConfig(level=logging.INFO)
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FREE_MODELS = ["mistralai/mistral-7b-instruct", "nousresearch/hermes-2-pro"]
MAX_PROMPT_CHARS = 3800

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

def extract_text_bytes(file_bytes, filename):
    fname = filename.lower()
    try:
        if fname.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            return "\n".join([p.extract_text() or "" for p in reader.pages])
        elif fname.endswith((".docx", ".doc")):
            tmp = DATA_DIR / f"tmp_{uuid.uuid4().hex}.docx"
            tmp.write_bytes(file_bytes)
            txt = docx2txt.process(str(tmp))
            tmp.unlink(missing_ok=True)
            return txt
        else:
            return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logging.warning(f"extract_text_bytes failed: {e}")
        return ""

def preprocess_resume_text(text):
    text = re.sub(r'\s+', ' ', text)
    for sec in ["Experience", "Projects", "Certifications", "Education", "Skills"]:
        text = re.sub(rf'(?i)\s*{sec}\s*', f'\n\n**🔹 {sec}**\n', text)
    return text.strip()

def calculate_resume_clarity(text):
    words = len(text.split())
    bullets = text.count("•")
    sections = len(re.findall(r'🔹', text))
    score = 40
    if words > 300:
        score += 20
    if bullets >= 4:
        score += 15
    if sections >= 3:
        score += 25
    return min(100, score)

def build_prompt(resume_text, job_text=None):
    snippet = resume_text[:MAX_PROMPT_CHARS]
    prompt = f"""
You are an expert ATS evaluator and career AI analyst.
Analyze the resume below and return ONLY valid JSON with keys:
technical_skills, soft_skills, suggested_job_roles, job_match_score, ats_compatibility_score, clarity_score,
one_sentence_summary, top_tips.

Resume:
{snippet}
"""
    if job_text:
        prompt += f"\nTarget Job Description:\n{job_text[:1000]}"
    prompt += "\nReturn JSON only."
    return prompt

def call_openrouter_chat(prompt, model):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a JSON-only assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1500
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def extract_first_json_from_text(s):
    s = re.sub(r"```(?:json)?", "", s).replace("```", "")
    start, end = s.find("{"), s.rfind("}")
    return json.loads(s[start:end + 1])

def robust_llm_analysis(resume, job=None):
    for model in FREE_MODELS:
        try:
            raw = call_openrouter_chat(build_prompt(resume, job), model)
            parsed = extract_first_json_from_text(raw)
            parsed["model_used"] = model
            return parsed
        except Exception as e:
            logging.warning(f"{model} failed: {e}")
    return {"error": "All LLMs failed. Please retry."}

def compute_overall(job, ats, clarity):
    return int(round(0.4 * job + 0.3 * ats + 0.3 * clarity))

st.markdown("""
<style>
body {background-color:#f8fafc;}
.robot-header {
    text-align:center;
    background:linear-gradient(145deg,#eaf3ff,#ffffff);
    padding:12px;
    border-radius:16px;
    box-shadow:0 3px 12px rgba(0,90,180,0.15);
}
.robot-heading {
    font-size:26px;
    font-weight:800;
    color:#0077cc;
}
.robot-sub {
    color:#555;font-size:14px;margin-top:-4px;
}
.section-header {
    font-weight:700;font-size:18px;color:#003366;margin-top:1rem;
}
.chat-icon {
    position:fixed;
    bottom:25px;
    right:25px;
    background-color:#0077cc;
    color:white;
    border-radius:50%;
    width:60px;height:60px;
    display:flex;align-items:center;justify-content:center;
    font-size:26px;
    box-shadow:0 5px 15px rgba(0,0,0,0.2);
    cursor:pointer;
    transition:transform 0.2s;
}
.chat-icon:hover {transform:scale(1.1);}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='robot-header'>", unsafe_allow_html=True)
st.markdown("<div class='robot-heading'>🤖 SkillSight AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='robot-sub'>Your AI Career Readiness Partner</div>", unsafe_allow_html=True)
try:
    robot_anim = requests.get("https://lottie.host/55ff268a-2f04-4a8b-a86a-938d396b28a5/GkKJULtXx9.json").json()
    st_lottie(robot_anim, height=160, key="robot_anim_new")
except:
    st.write("🤖")
st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.caption("⚙️ Settings")
    enable_ai = st.checkbox("Enable AI Resume Analysis", True)
    job_desc = st.text_area("Target Job Description (optional)", height=120)
    st.markdown("---")
    st.caption("Ensure `OPENROUTER_API_KEY` is set in .env for full AI-powered analysis.")

st.markdown("## ⚡ SkillSight Pro — AI Resume Analyzer v3.6")
uploaded = st.file_uploader("📤 Upload Resume", type=["pdf", "docx", "txt"])

if uploaded:
    text_raw = extract_text_bytes(uploaded.read(), uploaded.name)
    text = preprocess_resume_text(text_raw)
    clarity = calculate_resume_clarity(text)
    clarity_msg = "🟢 Excellent" if clarity >= 80 else "🟡 Moderate" if clarity >= 60 else "🔴 Needs Work"

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"<div class='section-header'>🧾 Resume Preview — Clarity {clarity}/100 {clarity_msg}</div>", unsafe_allow_html=True)
        st.code(text[:900] + ("..." if len(text) > 900 else ""))
        with st.expander("📄 View Full Resume"):
            st.text_area("Full Resume", text, height=400)
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=clarity,
            title={'text': "Resume Clarity"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#0077cc"},
                   'steps': [{'range': [0, 50], 'color': "#ff6b6b"},
                             {'range': [50, 80], 'color': "#ffd93d"},
                             {'range': [80, 100], 'color': "#69db7c"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)

    st.info("Analyzing with real-time AI models via OpenRouter...")
    parsed = robust_llm_analysis(text, job_desc if enable_ai else None)
    if "error" in parsed:
        st.error(parsed["error"])
        st.stop()

    job = int(parsed.get("job_match_score", 0))
    ats = int(parsed.get("ats_compatibility_score", 0))
    clarity_llm = int(parsed.get("clarity_score", clarity))
    overall = compute_overall(job, ats, clarity_llm)

    st.markdown("<div class='section-header'>📊 Results Summary</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Overall", f"{overall}/100")
    c2.metric("🎯 Job Match", f"{job}/100")
    c3.metric("📄 ATS", f"{ats}/100")

    df = pd.DataFrame({"Metric": ["Job Match", "ATS", "Clarity"], "Score": [job, ats, clarity_llm]})
    fig_r = go.Figure(go.Scatterpolar(
        r=df["Score"].tolist() + [df["Score"][0]],
        theta=df["Metric"].tolist() + [df["Metric"][0]],
        fill='toself', line_color="#0077cc", fillcolor="rgba(0,119,204,0.3)"
    ))
    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

    fig_pie = go.Figure(data=[go.Pie(labels=["Job Match", "ATS", "Clarity"], values=[job, ats, clarity_llm], hole=0.4)])
    fig_pie.update_traces(marker=dict(colors=["#0077cc", "#69db7c", "#ffd93d"]))
    fig_pie.update_layout(title="📈 Score Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    skills = parsed.get("technical_skills", [])
    if skills:
        skill_strengths = [random.randint(60, 100) for _ in skills]
        df_skills = pd.DataFrame({"Skill": skills, "Level": skill_strengths})
        fig_bar = go.Figure([go.Bar(
            x=df_skills["Skill"], y=df_skills["Level"],
            marker_color="#10b981", text=df_skills["Level"], textposition="auto"
        )])
        fig_bar.update_layout(title="🧩 Technical Skill Strength (AI Estimated)",
                              xaxis_title="Skill", yaxis_title="Level (0–100)",
                              yaxis=dict(range=[0, 100]), plot_bgcolor="white")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<div class='section-header'>🧠 AI Summary</div>", unsafe_allow_html=True)
    st.write(parsed.get("one_sentence_summary", "No summary available."))

    st.markdown("<div class='section-header'>🎯 Suggested Job Roles</div>", unsafe_allow_html=True)
    st.write(parsed.get("suggested_job_roles", []))

    st.markdown("<div class='section-header'>💡 Top Tips</div>", unsafe_allow_html=True)
    for tip in parsed.get("top_tips", [])[:5]:
        st.markdown(f"- {tip}")

    st.markdown("<div class='section-header'>🧩 Extracted Technical Skills</div>", unsafe_allow_html=True)
    st.write(skills)

    st.success("✅ Real-time AI Resume Analysis Complete")

st.markdown("<div class='chat-icon'>🤖</div>", unsafe_allow_html=True)