import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="Samvidhan AI", page_icon="🇳🇵", layout="centered")

# Custom CSS for better readability
st.markdown("""
<style>
    .legal-text { font-family: 'Georgia', serif; font-size: 1.1rem; line-height: 1.6; color: #2c3e50; background-color: #f9f9f9; padding: 15px; border-left: 5px solid #d35400; border-radius: 5px; }
    .highlight { background-color: #fff3cd; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🇳🇵 Samvidhan AI")
st.caption("Nepal's Constitution (2015) • Intelligent Legal Search")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask a question (e.g. 'How is the Prime Minister elected?')..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting the Constitution..."):
            try:
                response = requests.post(API_URL, json={"text": prompt})
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data["found"]:
                        msg = " I couldn't find specific articles matching that query."
                        st.markdown(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                    else:
                        # Intro Message
                        intro = f"I found **{len(data['results'])} articles** related to your query."
                        st.markdown(intro)
                        
                        # Accumulate text for chat history
                        history_text = intro + "\n\n"

                        # Display Results
                        for res in data['results']:
                            score = res['relevance'] * 100
                            ctx = res['graph_context']
                            
                            # Card UI
                            with st.expander(f" {res['article_id']} | {res['title']} ({score:.0f}% Match)", expanded=True):
                                # Legal Text Block
                                st.markdown(f'<div class="legal-text">{res["text"]}</div>', unsafe_allow_html=True)
                                st.write("") # Spacer
                                
                                # Metadata Badges
                                c1, c2, c3 = st.columns(3)
                                if ctx.get('rights'):
                                    c1.info(f"**Rights:**\n\n" + ", ".join(ctx['rights']))
                                if ctx.get('institutions'):
                                    c2.warning(f"**Authority:**\n\n" + ", ".join(ctx['institutions']))
                                if ctx.get('references'):
                                    c3.success(f"**Referenced:**\n\n" + ", ".join(ctx['references'][:3]))
                            
                            # Append to history text (simplified for storage)
                            history_text += f"**{res['article_id']}**: {res['text'][:200]}...\n\n"

                        st.session_state.messages.append({"role": "assistant", "content": history_text})
                else:
                    st.error("Server Error")
            except Exception as e:
                st.error(f"Connection Failed: {e}")