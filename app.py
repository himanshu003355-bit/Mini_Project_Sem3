import streamlit as st
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# ==============================================================================
# 1. NLP & MACHINE LEARNING MODEL PIPELINE
# ==============================================================================
@st.cache_resource
def get_trained_model():
    """
    Initializes and trains the baseline NLP feature extractor and classifier.
    Combines authentic SMS datasets with modern threat samples (KYC, credential theft,
    electricity bill scams, lottery phishing, and deceptive job offers).
    """
    dataset = [
        # --- BENIGN / GENUINE COMMUNICATIONS (Label: 0) ---
        ("Hey, are we still meeting for the project discussion today at 4:00 PM?", 0),
        ("Please find attached the final lab report for the semester evaluation.", 0),
        ("Can you pick up groceries on your way back home? Call mom when done.", 0),
        ("Reminder: Submit your internal assignment before midnight today.", 0),
        ("Could you please share the lecture notes for Unit 3?", 0),
        ("Are we having dinner together? The team is waiting in the cafeteria.", 0),
        ("Your Swiggy order has been delivered. Thank you for using our service.", 0),
        ("Dear student, your semester tuition receipt has been generated successfully.", 0),
        ("Are you attending the university seminar scheduled for tomorrow morning?", 0),
        ("Happy Birthday! Wishing you a fantastic year ahead filled with success.", 0),
        ("Your monthly account e-statement ending in 4921 is now available for download.", 0),
        ("Flight SG-817 is on time. Gate 4 boarding commences at 18:00 hrs.", 0),

        # --- MALICIOUS / SCAM / FRAUD COMMUNICATIONS (Label: 1) ---
        ("URGENT: Your SBI Bank account has been suspended! Click http://sbi-kyc-verify.cc to update PAN immediately.", 1),
        ("Congratulations! You won Rs 25,00,000 in KBC lottery. Call 9876543210 to claim your cash reward now.", 1),
        ("Dear Customer, Your electricity connection will be disconnected tonight at 9:30 PM due to an unpaid bill. Call 8901234567.", 1),
        ("Confidential: 492019 is your secret OTP for transaction. If not initiated by you, visit http://secure-bank.org/cancel.", 1),
        ("Work from home opportunity: Earn Rs 3,000 to 5,000 daily by watching short videos. Send WhatsApp resume to 9123456780.", 1),
        ("Aapka Bijli Bill update nahi hua hai, turant call kare electricity officer ko nahi to power cut ho jayegi.", 1),
        ("Security Notice: Your SIM card will be deactivated within 2 hours. Update Aadhaar KYC now at bit.ly/sim-update-kyc.", 1),
        ("Income Tax refund of Rs 14,850 has been approved. Confirm your banking details to receive credit: http://it-refund.in", 1),
        ("Free mobile recharge of Rs 599 available for all users. Limited period offer, click now to activate.", 1),
        ("Fraud Alert: Your credit card was debited Rs 49,999. If not authorized, visit http://dispute-txn.com to reverse immediately.", 1),
        ("Paytm KYC has expired! Click http://paytm-kyc-now.site and provide debit card details to avoid service disruption.", 1),
        ("You have an unclaimed parcel held at the distribution center. Confirm your address at http://courier-reschedule.xyz", 1)
    ]
    df = pd.DataFrame(dataset, columns=["text", "label"])

    # NLP Vectorization: Sub-linear TF scaling and word n-grams (1, 2)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=2500,
        sublinear_tf=True,
        lowercase=True
    )
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    # Supervised Learning Classifier: Linear Support Vector Classifier
    classifier = LinearSVC(C=1.2, random_state=42)
    classifier.fit(X, y)

    return vectorizer, classifier


vectorizer, model = get_trained_model()


# ==============================================================================
# 2. INFERENCE & EXPLAINABLE AI (XAI) REASONING ENGINE
# ==============================================================================
def analyze_scam_message(text: str):
    """
    Executes composite threat assessment:
    1. Heuristic & behavioral indicator parsing (Urgency, PII harvesting, Unverified URLs, Financial lures)
    2. Statistical NLP classifier decision boundary projection
    3. Multitier risk scoring and rule-to-text explainability generation
    """
    cleaned = text.lower().strip()
    flags = []
    threat_score = 0.0

    # 1. Psychological Triggers: Urgency, Intimidation & Coercion
    urgency_keywords = [
        "urgent", "immediately", "immediate", "within", "hours", "tonight", "suspended",
        "blocked", "deactivated", "cut", "disconnected", "last chance", "final warning",
        "action required", "freeze", "turant"
    ]
    matched_urgency = [w for w in urgency_keywords if re.search(r"\b" + re.escape(w) + r"\b", cleaned)]
    if matched_urgency:
        threat_score += 0.35
        flags.append(f"Artificial urgency and coercive language detected ({', '.join(matched_urgency[:3])}).")

    # 2. Cyber Threat Indicators: Suspicious TLDs and Shortened URLs
    url_pattern = re.findall(r"(https?://\S+|bit\.ly/\S+|www\.\S+|\S+\.(?:xyz|top|site|cc|in|co)/\S*)", cleaned)
    if url_pattern:
        threat_score += 0.40
        flags.append(f"Unverified external hyperlink or shortened URL identified: {url_pattern[0]}")

    # 3. Credential Harvesting: High-Value Sensitive PII Requests
    credential_keywords = [
        "otp", "password", "pin", "cvv", "pan card", "aadhaar", "bank account",
        "netbanking", "credit card", "debit card", "kyc", "verify identity"
    ]
    matched_creds = [w for w in credential_keywords if re.search(r"\b" + re.escape(w) + r"\b", cleaned)]
    if matched_creds:
        threat_score += 0.40
        flags.append(f"Solicitation of confidential authentication credentials or PII ({', '.join(matched_creds[:3])}).")

    # 4. Social Engineering: Unrealistic Financial Incentives & Lures
    lure_keywords = [
        "lottery", "won", "winner", "prize", "cashback", "reward", "free recharge",
        "earn daily", "part-time job", "refund approved", "crorepati", "congratulations"
    ]
    matched_lures = [w for w in lure_keywords if re.search(r"\b" + re.escape(w) + r"\b", cleaned)]
    if matched_lures:
        threat_score += 0.35
        flags.append(f"Unrealistic financial inducement or deceptive promotional hook detected ({', '.join(matched_lures[:3])}).")

    # 5. Machine Learning Inference
    vec_input = vectorizer.transform([cleaned])
    decision_margin = model.decision_function(vec_input)[0]
    ml_probability = 1.0 / (1.0 + np.exp(-decision_margin))

    # Composite Scoring Formulation (Weighted Heuristic Density + Model Confidence)
    composite_risk = min(1.0, (0.55 * threat_score) + (0.45 * ml_probability))

    # Tri-Tier Risk Stratification
    if composite_risk >= 0.52 or len(flags) >= 2:
        risk_label = "Scam"
        risk_category = "CRITICAL / HIGH THREAT"
    elif composite_risk >= 0.24 or len(flags) == 1:
        risk_label = "Suspicious"
        risk_category = "ELEVATED / MODERATE RISK"
    else:
        risk_label = "Genuine"
        risk_category = "LOW RISK / SAFE"

    return {
        "risk_label": risk_label,
        "risk_category": risk_category,
        "confidence_score": float(composite_risk),
        "flags": flags
    }


# ==============================================================================
# 3. STREAMLIT ENTERPRISE UI
# ==============================================================================
st.set_page_config(
    page_title="ScamGuard | AI-Powered Fraud Message Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# Enterprise Modern UI Styling
st.markdown("""
    <style>
    .header-container {
        padding: 0.5rem 0 1.2rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        margin-top: 0.3rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Top Application Banner
st.markdown("""
    <div class="header-container">
        <div class="main-title">🛡️ ScamGuard Enterprise Threat Intelligence</div>
        <div class="main-subtitle">NLP-driven multi-tier fraud message classification with Explainable Artificial Intelligence (XAI)</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration & Telemetry
with st.sidebar:
    st.subheader("System Specifications")
    st.markdown("""
    - **Architecture:** Hybrid NLP + ML Heuristics
    - **Feature Space:** TF-IDF Sub-linear $N$-grams
    - **Classification Model:** Linear Support Vector Machine (LinearSVC)
    - **Explainability:** Transparent Deterministic XAI
    - **Supported Channels:** SMS, WhatsApp, RCS, Chat Inboxes
    """)
    st.divider()

    st.subheader("Benchmark Test Cases")
    preset_selection = st.radio(
        "Select an automated test scenario:",
        [
            "Custom Input (Manual Inspection)",
            "Bank Impersonation & Phishing (High Risk)",
            "Utility Disconnection Threat (Urgency / Scam)",
            "Deceptive Work-From-Home Offer (Suspicious)",
            "Standard Operational Meeting (Genuine)"
        ]
    )

preset_library = {
    "Bank Impersonation & Phishing (High Risk)": 
        "URGENT: Your SBI Bank account has been suspended due to pending PAN KYC. Click http://sbi-kyc-verify.cc to update immediately.",
    "Utility Disconnection Threat (Urgency / Scam)": 
        "Notice: Your electricity connection will be disconnected tonight at 9:30 PM due to an unpaid bill. Call the electricity officer at 9876543210 immediately.",
    "Deceptive Work-From-Home Offer (Suspicious)": 
        "Congratulations! You are shortlisted for an online position. Earn Rs 5,000 daily from home just by reviewing videos. Contact HR on WhatsApp now.",
    "Standard Operational Meeting (Genuine)": 
        "Hi team, are we still meeting in conference room 3 at 4:00 PM today to finalize the technical documentation?"
}

default_content = "" if preset_selection == "Custom Input (Manual Inspection)" else preset_library[preset_selection]

# Main Workspace
st.markdown("### Message Inspection Console")
user_message = st.text_area(
    "Paste SMS, messaging app text, or communication string to evaluate:",
    value=default_content,
    height=130,
    placeholder="e.g., Immediate action required: Your account has been temporarily restricted. Verify your credentials at https://..."
)

col_eval, col_reset, _ = st.columns([1.5, 1, 6])
with col_eval:
    submit_scan = st.button("Evaluate Message", type="primary", use_container_width=True)
with col_reset:
    if st.button("Clear Field", use_container_width=True):
        st.rerun()

# Execution & Threat Reporting
if submit_scan:
    if not user_message.strip():
        st.warning("Input required: Please provide text content to perform threat analysis.")
    else:
        analysis_result = analyze_scam_message(user_message)
        label = analysis_result["risk_label"]
        category = analysis_result["risk_category"]
        risk_percentage = analysis_result["confidence_score"] * 100
        flags = analysis_result["flags"]

        st.divider()
        st.markdown("### Threat Intelligence Report")

        metric_col, explanation_col = st.columns([1, 1.8], gap="large")

        with metric_col:
            st.markdown("#### Classification Verdict")
            if label == "Scam":
                st.error(f"### 🚨 {label.upper()}")
                st.metric(label="Calculated Threat Score", value=f"{risk_percentage:.1f}%", delta="Critical Risk Level", delta_color="inverse")
            elif label == "Suspicious":
                st.warning(f"### ⚠️ {label.upper()}")
                st.metric(label="Calculated Threat Score", value=f"{risk_percentage:.1f}%", delta="Elevated Risk Level", delta_color="off")
            else:
                st.success(f"### ✅ {label.upper()}")
                st.metric(label="Calculated Threat Score", value=f"{risk_percentage:.1f}%", delta="Benign Communication", delta_color="normal")

            st.write("**Threat Exposure Meter:**")
            st.progress(analysis_result["confidence_score"])
            st.caption(f"Risk Rating: **{category}**")

        with explanation_col:
            st.markdown("#### Explainable AI (XAI) Threat Breakdown")
            if flags:
                for item in flags:
                    st.markdown(f"- 🔴 **Identified Indicator:** {item}")
                st.info("System Recommendation: Do not interact with links, disclose credentials, or respond to this communication.")
            else:
                st.markdown("- 🟢 **Clean:** No psychological intimidation, deceptive links, or credential harvesting patterns identified.")
                st.caption("System Recommendation: Communication is consistent with standard benign messaging patterns.")

st.markdown("---")
st.caption("ScamGuard AI Platform • Natural Language Processing & Security Analytics")


