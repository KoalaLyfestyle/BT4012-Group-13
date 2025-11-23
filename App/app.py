import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import time

# Add current directory to path so we can import local modules
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import get_available_models, load_model, fetch_phishing_urls, SafeTruncatedSVD, check_gpu_status, get_feature_columns
from feature_extractor import extract_url_features

# Hack for unpickling SafeTruncatedSVD if it was saved as __main__.SafeTruncatedSVD
if '__main__' in sys.modules:
    sys.modules['__main__'].SafeTruncatedSVD = SafeTruncatedSVD

st.set_page_config(
    page_title="Phishing Detection Lab",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #FAFAFA;
        border: 1px solid #4B5563;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Metric Cards with Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif;
    }
    .metric-label {
        color: #9CA3AF;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2em;
        font-weight: 700;
        margin: 0;
    }
    
    /* Custom Warning Box */
    .warning-box {
        background-color: rgba(255, 193, 7, 0.1);
        color: #FFC107;
        padding: 16px;
        border-radius: 8px;
        border: 1px solid rgba(255, 193, 7, 0.2);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Phishing Detection Lab")
st.markdown("### Advanced Threat Detection System")

# Sidebar
st.sidebar.header("Configuration")

# Force Stop / Reset
if st.sidebar.button("Force Stop / Reset", help="Stops execution and clears cache"):
    st.cache_resource.clear()
    st.rerun()

# System Status (Lazy Check)
with st.sidebar.expander("System Status"):
    if st.button("Check GPU Availability"):
        with st.spinner("Checking system compatibility..."):
            gpu_status = check_gpu_status()
        
        if gpu_status["available"]:
            st.success(f"Acceleration: {gpu_status['device']}")
            for detail in gpu_status["details"]:
                st.caption(detail)
        else:
            st.warning("No GPU detected (CPU only)")
            st.caption("Neural Networks may be slower.")

mode = st.sidebar.radio("Select Mode", ["Single URL Analysis", "Batch API Testing"])

# Load Models
@st.cache_resource
def get_models_list():
    return get_available_models()

with st.spinner("Scanning for available models..."):
    available_models = get_models_list()

if not available_models:
    st.error("No models found in the experiments directory!")
    st.stop()

# Helper to run inference
def run_inference(model, url, model_info, precomputed_features=None):
    # Create DataFrame if features not provided
    if precomputed_features is None:
        df = pd.DataFrame({'url': [url]})
        # Feature Engineering
        try:
            processed_df = extract_url_features(df)
        except Exception as e:
            return None, f"Feature Extraction Error: {str(e)}"
    else:
        processed_df = precomputed_features.copy()

    
    # Pre-processing based on model type
    vectorizer_type = model_info.get("vectorizer", "")
    if "Numeric" in vectorizer_type:
        # Get all non-numeric columns dynamically
        string_cols = processed_df.select_dtypes(exclude=[np.number]).columns.tolist()
        processed_df = processed_df.drop(columns=string_cols)
        
        # Align features with model expectations (critical for numeric models)
        try:
            expected_features = get_feature_columns(model_info["id"])
            if expected_features:
                # Reindex to match expected features, filling missing with 0
                processed_df = processed_df.reindex(columns=expected_features, fill_value=0)
        except Exception:
            # If feature alignment fails, continue with current features
            pass


    # Prediction
    try:
        # Check if model is a pipeline (sklearn)
        if hasattr(model, "predict_proba"):
            # Special handling for pure TF-IDF pipelines that expect 1D input (list of strings)
            is_pure_tfidf = False
            if hasattr(model, "steps"):
                first_step = model.steps[0][1]
                if first_step.__class__.__name__ == "TfidfVectorizer":
                    is_pure_tfidf = True
            
            if is_pure_tfidf:
                # Pass the raw URL as a list (1D array-like)
                # If input is a single string, wrap it. If it's a list/series, pass as is.
                if isinstance(url, str):
                    probs = model.predict_proba([url])[:, 1]
                    return probs[0], None
                else:
                    # Batch mode for TF-IDF
                    probs = model.predict_proba(url)[:, 1]
                    return probs, None
            else:
                probs = model.predict_proba(processed_df)[:, 1]
                if len(probs) == 1:
                    return probs[0], None
                return probs, None
            
        # Check if model is Keras
        elif hasattr(model, "predict"):
            try:
                probs = model.predict(processed_df)
                if len(probs) == 1:
                    return float(probs[0][0]), None
                return probs.flatten(), None
            except:
                return None, "Incompatible Model Input"
        else:
            return None, "Unknown Model Type"
    except Exception as e:
        return None, f"Inference Error: {str(e)}"


if mode == "Single URL Analysis":
    st.subheader("Single URL Analysis")
    
    # Model Selection
    model_options = {m["name"]: m for m in available_models}
    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
    selected_model_info = model_options[selected_model_name]
    
    url_input = st.text_input("Enter URL to analyze", placeholder="http://example.com")
    
    if st.button("Analyze URL"):
        if not url_input:
            st.warning("Please enter a URL.")
        else:
            with st.spinner(f"Loading {selected_model_name}..."):
                try:
                    model = load_model(selected_model_info["id"])
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    st.stop()
            
            with st.spinner("Analyzing..."):
                prob, error = run_inference(model, url_input, selected_model_info)
                
                if error:
                    st.error(error)
                else:
                    # Display Result
                    col1, col2 = st.columns(2)
                    
                    is_phishing = prob > 0.5
                    status_color = "#EF4444" if is_phishing else "#10B981" # Red-500 or Green-500
                    status_text = "PHISHING DETECTED" if is_phishing else "LEGITIMATE"
                    status_icon = "🚨" if is_phishing else "✅"
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Verdict</div>
                            <div class="metric-value" style="color: {status_color};">
                                {status_icon} {status_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence Score</div>
                            <div class="metric-value" style="color: #FAFAFA;">
                                {prob:.2%}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gauge Chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Phishing Probability", 'font': {'size': 24, 'color': "#FAFAFA"}},
                        number = {'suffix': "%", 'font': {'color': "#FAFAFA"}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#FAFAFA"},
                            'bar': {'color': status_color},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#333",
                            'steps': [
                                {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.3)"},
                                {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
                            'threshold': {
                                'line': {'color': "white", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}}))
                    
                    fig.update_layout(
                        paper_bgcolor = 'rgba(0,0,0,0)',
                        plot_bgcolor = 'rgba(0,0,0,0)',
                        font = {'color': "#FAFAFA"},
                        height = 300,
                        margin = dict(l=20, r=20, t=50, b=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature Breakdown (Optional - if we want to show extracted features)
                    with st.expander("View Extracted Features"):
                        df = pd.DataFrame({'url': [url_input]})
                        feats = extract_url_features(df)
                        # Transpose and convert to string to avoid Arrow type errors with mixed types in one column
                        st.dataframe(feats.T.astype(str))

elif mode == "Batch API Testing":
    st.subheader("Batch API Testing (OpenPhish Feed)")
    
    # Settings
    col1, col2 = st.columns([3, 1])
    with col1:
        # Select Models
        model_options = {m["name"]: m for m in available_models}
        # Default to top 3 by AUC
        default_models = list(model_options.keys())[:3]
        selected_models = st.multiselect("Select Models to Compare", list(model_options.keys()), default=default_models)
    
    with col2:
        limit = st.number_input("Max URLs", min_value=5, max_value=1000, value=20)
    
    if st.button("Fetch URLs & Run Benchmark"):
        if not selected_models:
            st.warning("Select at least one model.")
        else:
            # 1. Fetch URLs
            with st.spinner("Fetching live phishing URLs from OpenPhish..."):
                urls = fetch_phishing_urls(limit=limit)
            
            if not urls:
                st.error("Failed to fetch URLs or feed is empty.")
            else:
                st.success(f"Successfully fetched {len(urls)} URLs.")
                
                # 2. Load Models
                loaded_models = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, m_name in enumerate(selected_models):
                    status_text.text(f"Loading model: {m_name}...")
                    try:
                        m_info = model_options[m_name]
                        loaded_models[m_name] = load_model(m_info["id"])
                    except Exception as e:
                        st.error(f"Could not load {m_name}: {e}")
                    progress_bar.progress((i + 1) / len(selected_models) * 0.3)
                
                # 3. Run Inference
                # Optimize: Extract features ONCE for all URLs
                status_text.text("Extracting features for all URLs...")
                try:
                    batch_df = pd.DataFrame({'url': urls})
                    batch_features = extract_url_features(batch_df)
                    
                    results = []
                    # Initialize results with URLs
                    for url in urls:
                        results.append({"URL": url})
                        
                    # Run inference model by model (batch processing)
                    for i, m_name in enumerate(selected_models):
                        status_text.text(f"Running inference for {m_name}...")
                        model = loaded_models.get(m_name)
                        if not model:
                            continue
                            
                        m_info = model_options[m_name]
                        
                        # Pass the full batch to run_inference
                        # For TF-IDF, we pass the list of URLs. For others, the precomputed features.
                        try:
                            probs, err = run_inference(model, urls, m_info, precomputed_features=batch_features)
                            
                            if err:
                                # If batch fails, try single? Or just mark error
                                for row in results:
                                    row[m_name] = None
                            else:
                                # Assign probabilities to results
                                for idx, prob in enumerate(probs):
                                    results[idx][m_name] = prob
                        except Exception as e:
                             for row in results:
                                    row[m_name] = None
                        
                        progress_bar.progress(0.3 + ((i + 1) / len(selected_models)) * 0.7)

                except Exception as e:
                    st.error(f"Batch processing failed: {e}")
                
                status_text.text("Done!")
                progress_bar.empty()
                
                # 4. Display Results
                results_df = pd.DataFrame(results)
                
                st.markdown("### Benchmark Results")
                st.dataframe(results_df)
                
                # 5. Comparison Dashboard
                st.markdown("### Model Comparison")
                
                # Calculate Detection Rate (assuming all fetched URLs are phishing)
                metrics = []
                for m_name in loaded_models.keys():
                    # Count how many > 0.5
                    detected = results_df[m_name].apply(lambda x: 1 if x is not None and x > 0.5 else 0).sum()
                    rate = detected / len(urls)
                    metrics.append({"Model": m_name, "Detection Rate": rate})
                
                metrics_df = pd.DataFrame(metrics)
                
                # Bar Chart
                fig = px.bar(
                    metrics_df, 
                    x="Model", 
                    y="Detection Rate", 
                    color="Detection Rate",
                    range_y=[0, 1],
                    title="Phishing Detection Rate (Recall) on Live Feed",
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Stats
                st.markdown("#### Detailed Statistics")
                st.table(metrics_df.style.format({"Detection Rate": "{:.2%}"}))

