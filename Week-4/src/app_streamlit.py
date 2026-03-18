import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Week 4 AI", page_icon="🎬")
st.title("🎬 Sentiment Analysis AI")
st.caption("Powered by Qwen-0.5B + LoRA")

review_text = st.text_area("Enter Movie Review:", height=150)

if st.button("Analyze", type="primary"):
    if not review_text.strip():
        st.warning("Please enter text.")
    else:
        try:
            with st.spinner("Analyzing..."):
                response = requests.post(API_URL, json={"text": review_text})
                
            if response.status_code == 200:
                data = response.json()
                label = data["label"]
                conf = data["confidence"]
                latency = data["latency"]
                
                # Display Metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", label)
                col2.metric("Confidence", f"{conf:.2%}")
                col3.metric("Latency", f"{latency}s")
                
                # Visual Bar
                if label == "Positive":
                    st.success(f"This review is **POSITIVE** ({conf:.1%} sure)")
                    st.progress(conf)
                else:
                    st.error(f"This review is **NEGATIVE** ({conf:.1%} sure)")
                    st.progress(conf)
            else:
                st.error("API Error")
                
        except Exception as e:
            st.error(f"Connection Failed: {e}")