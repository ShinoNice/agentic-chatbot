# ⚠️ DEPRECATED — This file is kept for reference only.
# Please use ui/streamlit_frontend.py instead, which is the
# actively maintained frontend for the Agentic RAG Chatbot.

import streamlit as st
import httpx
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Research Agent",
                   page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent Hybrid RAG")
st.markdown("---")

# --- 2. Sidebar & Settings ---
with st.sidebar:
    st.header("Connection Settings")
    # Point this to your FastAPI URL
    api_url = st.text_input(
        "API Gateway", value="http://127.0.0.1:8000/api/chat")
    st.info("Ensure the FastAPI server is running in your terminal.")

# --- 3. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            st.caption(f"Sources: {', '.join(message['sources'])}")

# --- 4. User Input ---
if prompt := st.chat_input("Ask a question about Google or DeepSeek reports..."):
    # Add user message to UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call the API
    with st.chat_message("assistant"):
        with st.spinner("The Agentic Team is researching..."):
            try:
                # We use httpx to talk to our FastAPI Backend
                payload = {"question": prompt}
                response = httpx.post(api_url, json=payload, timeout=90.0)

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    verification = data.get("verification")

                    # Display Answer
                    st.markdown(answer)

                    # Display Verification Badge
                    if verification is not None:
                        status = "Verified" if verification.get(
                            "supported") else "Unverified"
                        badge_color = "green" if verification.get(
                            "supported") else "orange"
                    else:
                        status = "Not checked"
                        badge_color = "orange"
                    st.markdown(f"**Status:** :{badge_color}[{status}]")

                    if sources:
                        source_names = [s.get("source", "unknown")
                                        for s in sources]
                        st.caption(f"📚 Sources: {', '.join(source_names)}")

                    # Save to state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": [s.get("source", "unknown") for s in sources]
                    })
                else:
                    st.error(
                        f"API Error ({response.status_code}): {response.text}")

            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")


# import streamlit as st
# import httpx
# import json
# import time

# st.set_page_config(page_title="AI Research Agent", page_icon="🕵️", layout="wide")

# st.title("🕵️ Multi-Agent Research Portal")
# st.markdown("---")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display History (Corrected to show persistent status logs)
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         if "logs" in message and message["logs"]:
#             with st.expander("📝 View Agent Reasoning", expanded=False):
#                 for log in message["logs"]:
#                     st.write(log)
#         st.markdown(message["content"])
#         if "metadata" in message:
#             st.caption(f"Sources: {', '.join(message['metadata']['sources'])}")

# if prompt := st.chat_input("Ask a question..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         # 1. Setup Placeholders
#         status_container = st.status("Team is coordinating...", expanded=True)
#         answer_placeholder = st.empty()

#         full_answer = ""
#         agent_logs = []  # To persist the 'empty dropdown' content

#         try:
#             with httpx.stream("POST", "http://127.0.0.1:8000/v1/chat/stream", json={"question": prompt}, timeout=None) as r:
#                 for line in r.iter_lines():
#                     if not line.startswith("data: "): continue

#                     data = json.loads(line[6:])

#                     # A. Agent Logs (The Dropdown Content)
#                     if data["type"] == "node":
#                         log_entry = f"⚙️ **{data['content'].replace('_', ' ').title()}** agent starting..."
#                         agent_logs.append(log_entry)
#                         status_container.write(log_entry)

#                     # B. Answer Tokens (with Smooth Delay)
#                     elif data["type"] == "token":
#                         full_answer += data["content"]
#                         answer_placeholder.markdown(full_answer + "▌")
#                         # --- SENIOR FIX: Slow down for readability ---
#                         time.sleep(0.01)

#                     # C. Final Metadata
#                     elif data["type"] == "metadata":
#                         status_container.update(label="✅ Analysis Complete", state="complete", expanded=False)
#                         answer_placeholder.markdown(full_answer) # Remove cursor

#                         st.session_state.messages.append({
#                             "role": "assistant",
#                             "content": full_answer,
#                             "logs": agent_logs, # Save logs so dropdown works on refresh
#                             "metadata": data
#                         })

#                         # Show sources immediately
#                         st.caption(f"Sources: {', '.join(data['sources'])}")

#         except Exception as e:
#             st.error(f"Backend Link Broken: {e}")
