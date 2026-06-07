import streamlit as st
import os, re, json, time, requests, tempfile
from pathlib import Path
from bs4 import BeautifulSoup
import fitz
import textstat
from fpdf import FPDF
from transformers import pipeline

st.set_page_config(
    page_title="PPIIA — Public Policy Insight & Impact Analyzer",
    page_icon="🏛️",
    layout="wide"
)

# ── Load models once (cached so they don't reload every run) ─────────────────
@st.cache_resource
def load_models():
    st.info("Loading AI models for the first time... this takes 2-3 minutes")
    summarizer   = pipeline("summarization",          model="facebook/bart-large-cnn")
    classifier   = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    st.success("Models loaded!")
    return summarizer, classifier

# ── Extractors ────────────────────────────────────────────────────────────────
def extract_pdf(path):
    doc  = fitz.open(path)
    text = "".join(p.get_text() for p in doc)
    doc.close()
    return text

def extract_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    resp    = requests.get(url, headers=headers, timeout=30)
    soup    = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","footer"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

def preprocess(text):
    text = re.sub(r"\s+",            " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"Page \d+ of \d+","", text)
    return text.strip()

# ── Chunker (BART has 1024 token limit) ──────────────────────────────────────
def chunk_text(text, max_chars=3000):
    words  = text.split()
    chunks = []
    current = []
    count   = 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= max_chars:
            chunks.append(" ".join(current))
            current = []
            count   = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

# ── Core analysis functions ───────────────────────────────────────────────────
def classify_topic(text, classifier):
    categories = [
        "Finance and Budget", "Agriculture and Farming",
        "Technology and Digital", "Healthcare and Medicine",
        "Education and Learning", "Environment and Climate",
        "Defence and Security", "Labour and Employment",
        "Infrastructure and Transport", "Social Welfare",
        "Legal and Justice", "Other"
    ]
    sample = text[:1000]
    result = classifier(sample, candidate_labels=categories)
    return result["labels"][0], round(result["scores"][0]*100, 1)


def summarize_text(text, summarizer):
    chunks   = chunk_text(text, max_chars=3000)
    summaries= []
    for chunk in chunks[:3]:   # use first 3 chunks max
        if len(chunk.split()) < 50:
            continue
        out = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(out[0]["summary_text"])
    return " ".join(summaries)


def extract_key_provisions(text):
    provisions = []
    lines = text.split(".")
    keywords = ["shall","must","provided","notwithstanding","hereby",
                "amend","insert","substitute","omit","penalty","fine",
                "license","register","authority","board","commission"]
    for line in lines:
        line = line.strip()
        if len(line) > 60 and any(k in line.lower() for k in keywords):
            provisions.append(line)
        if len(provisions) >= 8:
            break
    return provisions


def extract_chronology(text):
    acts      = re.findall(r'[A-Z][a-zA-Z\s]+Act[,\s]+\d{4}', text)
    acts      = list(set(acts))[:10]
    years     = re.findall(r'\b(19|20)\d{2}\b', text)
    years     = sorted(set(years))
    sections  = re.findall(r'[Ss]ection\s+\d+[A-Za-z]?\s+of\s+[A-Z][a-zA-Z\s]+Act', text)
    sections  = list(set(sections))[:5]
    return {
        "predecessor_acts": acts if acts else ["No previous acts found in document"],
        "years_referenced": years,
        "sections_amended": sections if sections else ["No specific sections found"]
    }


def detect_affected_sectors(text, classifier):
    sectors = [
        "Agriculture", "Finance and Banking", "Healthcare",
        "Education", "Technology", "Environment",
        "Real Estate", "Manufacturing", "Small Business",
        "Government Administration"
    ]
    sample = text[:1000]
    result = classifier(sample, candidate_labels=sectors, multi_label=True)
    affected = [
        {"sector": label, "score": round(score*100,1)}
        for label, score in zip(result["labels"], result["scores"])
        if score > 0.3
    ]
    return affected[:5]


def assess_impact(text, classifier):
    timeframes = {
        "Short-term (0-1 year)": [
            "immediate implementation", "compliance deadline",
            "enforcement", "notification", "rules framing"
        ],
        "Medium-term (1-5 years)": [
            "structural change", "industry adaptation",
            "market shift", "regulatory compliance", "reform"
        ],
        "Long-term (>5 years)": [
            "systemic change", "economic growth",
            "social transformation", "development", "sustainability"
        ]
    }
    results = {}
    for timeframe, keywords in timeframes.items():
        found = [k for k in keywords if k in text.lower()]
        if found:
            results[timeframe] = f"Key themes: {', '.join(found[:3])}"
        else:
            results[timeframe] = "Limited direct references found — indirect impact likely"
    return results


def detect_risks(text, classifier):
    risk_labels = [
        "high financial burden on citizens",
        "privacy and data concerns",
        "implementation challenges",
        "environmental risk",
        "social inequality",
        "low risk"
    ]
    sample = text[:1000]
    result = classifier(sample, candidate_labels=risk_labels, multi_label=True)
    risks  = [
        label for label, score in zip(result["labels"], result["scores"])
        if score > 0.25 and label != "low risk"
    ]
    level = "low"
    if len(risks) >= 3: level = "high"
    elif len(risks) >= 1: level = "medium"
    return {"risks": risks if risks else ["No major risks detected"], "level": level}


def make_pdf(data):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,12,"PPIIA - Bill Analysis Report",ln=True,align="C")
    pdf.set_font("Arial","B",11)
    pdf.cell(0,8,data.get("bill_name",""),ln=True,align="C")
    pdf.ln(4)
    for heading, body in [
        ("Bill Category",     data.get("topic","")),
        ("Summary",           data.get("summary","")),
        ("Key Provisions",    "\n".join(data.get("provisions",[]))),
        ("Affected Sectors",  "\n".join(s["sector"] for s in data.get("sectors",[]))),
        ("Risks",             "\n".join(data.get("risks",{}).get("risks",[]))),
    ]:
        pdf.set_font("Arial","B",11)
        pdf.set_fill_color(220,220,220)
        pdf.cell(0,8,heading,ln=True,fill=True)
        pdf.set_font("Arial","",10)
        pdf.multi_cell(0,6,str(body).encode("latin-1","replace").decode("latin-1"))
        pdf.ln(3)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("PPIIA - Public Policy Insight & Impact Analyzer")
st.markdown("#### Simplifying Government Bills for Every Citizen")
st.markdown("*No API key needed — runs on free open-source AI models*")
st.markdown("---")

with st.sidebar:
    st.title("About")
    st.markdown("""
    **Models used (all free):**
    - `facebook/bart-large-cnn` for summarization
    - `facebook/bart-large-mnli` for classification

    **Get Bill PDFs:**
    - [Sansad.in](https://sansad.in/ls/knowledge-centre/government-bills)
    - [PRS Legislative](https://prsindia.org/billtrack)
    """)
    st.caption("GUVI x HCL Capstone | CivicTech")

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Input")
    pdf_file = st.file_uploader("Upload Bill PDF", type=["pdf"])
    url_in   = st.text_input("Or paste Sansad URL", placeholder="https://sansad.in/...")
    analyze  = st.button("Analyze Bill", type="primary", use_container_width=True)

with col2:
    st.subheader("How to get a bill PDF")
    st.info("""
1. Go to sansad.in/ls/knowledge-centre/government-bills
2. Use Session and Year filters to find bills
3. Click any bill and download the PDF
4. Upload it on the left
    """)

st.markdown("---")

if analyze:
    if not pdf_file and not url_in.strip():
        st.error("Please upload a PDF or paste a URL.")
        st.stop()

    summarizer, classifier = load_models()

    progress = st.progress(0)
    status   = st.empty()

    status.info("Extracting text...")
    progress.progress(10)
    if pdf_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_file.read())
        tmp.close()
        raw       = extract_pdf(tmp.name)
        bill_name = pdf_file.name.replace(".pdf","")
    else:
        raw       = extract_url(url_in.strip())
        bill_name = "bill_from_url"

    cleaned = preprocess(raw)
    st.success(f"Extracted {len(cleaned):,} characters from {bill_name}")

    status.info("Classifying bill topic...")
    progress.progress(20)
    topic, confidence = classify_topic(cleaned, classifier)

    status.info("Summarizing bill...")
    progress.progress(35)
    summary = summarize_text(cleaned, summarizer)

    status.info("Extracting key provisions...")
    progress.progress(50)
    provisions = extract_key_provisions(cleaned)

    status.info("Building chronology...")
    progress.progress(62)
    chronology = extract_chronology(cleaned)

    status.info("Detecting affected sectors...")
    progress.progress(74)
    sectors = detect_affected_sectors(cleaned, classifier)

    status.info("Assessing impact timeframes...")
    progress.progress(84)
    impact = assess_impact(cleaned, classifier)

    status.info("Detecting risks...")
    progress.progress(94)
    risks = detect_risks(cleaned, classifier)

    progress.progress(100)
    status.success("Analysis complete!")

    st.markdown("---")
    st.markdown(f"## Analysis: {bill_name}")

    risk_color = {"low":"🟢","medium":"🟡","high":"🔴"}.get(risks["level"],"⚪")
    c1,c2,c3 = st.columns(3)
    c1.metric("Bill Category",    topic.split(" and ")[0])
    c2.metric("Confidence",       f"{confidence}%")
    c3.metric("Risk Level",       f"{risk_color} {risks['level'].upper()}")
    st.markdown("---")

    tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
        "Summary","Chronology","Sectors","Impact Timeframe","Risks","Key Provisions"
    ])

    with tab1:
        st.subheader("AI-Generated Summary")
        st.write(summary)
        st.markdown("---")
        score = textstat.flesch_reading_ease(summary)
        st.metric("Readability Score (Flesch)", f"{score:.0f}/100",
                  help="60+ means easy to read for general public")

    with tab2:
        st.subheader("Legislative History & Chronology")
        st.markdown("**Previous Acts Referenced:**")
        for a in chronology["predecessor_acts"]:
            st.markdown(f"- {a}")
        st.markdown("**Years Referenced in Bill:**")
        st.write(", ".join(chronology["years_referenced"]) or "None found")
        st.markdown("**Sections Amended:**")
        for s in chronology["sections_amended"]:
            st.markdown(f"- {s}")

    with tab3:
        st.subheader("Affected Sectors & Industries")
        if sectors:
            for s in sectors:
                st.progress(s["score"]/100, text=f"{s['sector']} — {s['score']}% relevance")
        else:
            st.write("No specific sectors detected")

    with tab4:
        st.subheader("Short / Medium / Long Term Impact")
        for timeframe, desc in impact.items():
            with st.expander(timeframe):
                st.write(desc)

    with tab5:
        st.subheader("Risks & Concerns")
        st.markdown(f"**Overall Risk Level: {risk_color} {risks['level'].upper()}**")
        st.markdown("**Identified Risks:**")
        for r in risks["risks"]:
            st.markdown(f"- {r}")

    with tab6:
        st.subheader("Key Provisions Extracted")
        for i, p in enumerate(provisions, 1):
            st.markdown(f"**{i}.** {p}")

    st.markdown("---")
    st.subheader("Download Results")
    data = {
        "bill_name": bill_name, "topic": topic,
        "summary": summary, "provisions": provisions,
        "chronology": chronology, "sectors": sectors,
        "impact": impact, "risks": risks
    }
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button("Download JSON Report",
            data=json.dumps(data, indent=2, ensure_ascii=False),
            file_name=f"{bill_name}_analysis.json",
            mime="application/json")
    with dl2:
        pdf_path = make_pdf(data)
        with open(pdf_path,"rb") as f:
            st.download_button("Download PDF Summary",
                data=f.read(),
                file_name=f"{bill_name}_summary.pdf",
                mime="application/pdf")

st.markdown("---")
st.caption("PPIIA | GUVI x HCL Capstone | CivicTech / Public Policy Analytics")
