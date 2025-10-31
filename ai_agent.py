# ---------------------------------------------------
#  AI CHATBOT AGENT WITH TOOLS (GROQ + Streamlit)
#  Tools:
#   - WebSearch (DuckDuckGo)
#   - Wikipedia
#   - Calculator
#   - Translator (English <-> Urdu)
# ---------------------------------------------------

import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia
from deep_translator import GoogleTranslator


# -----------------------------
# 1. ENVIRONMENT + STREAMLIT SETUP
# -----------------------------
load_dotenv()

st.set_page_config(page_title="AI Chatbot Agent", page_icon="💬")
st.title("💬 AI Chatbot Agent with Tools")

st.sidebar.header("Settings")
api_key = st.sidebar.text_input("GROQ API Key (starts with gsk_)", type="password") or os.getenv("GROQ_API_KEY", "")
model_name = st.sidebar.selectbox("Model", ["llama-3.1-8b-instant", "gemma2-9b-it"], index=0)
max_steps = st.sidebar.slider("Max reasoning steps", 1, 6, 3)

st.markdown("""
This chatbot can think and use real tools:
- 🌐 Web Search  
- 📚 Wikipedia  
- 🧮 Calculator  
- 🌍 Translator (English ↔ Urdu)
""")


# -----------------------------
# 2. TOOL FUNCTIONS
# -----------------------------
def tool_web_search(query, k=3):
    """Search using DuckDuckGo."""
    try:
        with DDGS() as ddg:
            results = ddg.text(query, region="us-en", max_results=k)
            lines = []
            for r in results:
                title, link, body = r.get("title", ""), r.get("href", ""), r.get("body", "")
                lines.append(f"- {title} — {link}\n  {body}")
            return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"WebSearch error: {e}"

def tool_wikipedia(query, sentences=2):
    """Fetch Wikipedia summary."""
    try:
        wikipedia.set_lang("en")
        pages = wikipedia.search(query, results=1)
        if not pages:
            return "No Wikipedia page found."
        summary = wikipedia.summary(pages[0], sentences=sentences)
        return f"Wikipedia: {pages[0]}\n{summary}"
    except Exception as e:
        return f"Wikipedia error: {e}"

def tool_calculator(expression):
    """Evaluate simple math expressions safely."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"

def tool_translate(text):
    """Translate English ↔ Urdu using Deep Translator."""
    try:
        lang = "ur" if re.search(r"[a-zA-Z]", text) else "en"
        translated = GoogleTranslator(source='auto', target=lang).translate(text)
        return f"Translation ({lang}): {translated}"
    except Exception as e:
        return f"Translation error: {e}"


# -----------------------------
# 3. SYSTEM PROMPT (Reasoning Template)
# -----------------------------
SYSTEM_PROMPT = """
You are an intelligent AI assistant with access to these tools:
1) WebSearch — use for recent or general info
2) Wikipedia — use for factual or background knowledge
3) Calculator — use for solving math or numeric queries
4) Translator — use for translating between English and Urdu

Follow this format:
Thought: what you think next
Action: which tool to use (WebSearch / Wikipedia / Calculator / Translator)
Action Input: what to search or compute
(Then you receive an Observation)

Repeat until ready, then write:
Final Answer: <clear helpful reply to the user>
"""

ACTION_RE = re.compile(r"^Action:\s*(WebSearch|Wikipedia|Calculator|Translator)", re.I)
INPUT_RE  = re.compile(r"^Action Input:\s*(.*)", re.I)


# -----------------------------
# 4. MAIN AGENT FUNCTION
# -----------------------------
def ai_agent(client, model, user_input, max_iters=3):
    transcript = [f"User: {user_input}"]
    observation = None

    for step in range(1, max_iters + 1):
        convo = SYSTEM_PROMPT + "\n" + "\n".join(transcript)
        if observation:
            convo += f"\nObservation: {observation}"

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": convo},
            ],
            temperature=0.3,
            max_tokens=512,
        )

        text = resp.choices[0].message.content or ""

        with st.expander(f"🧠 Step {step}", expanded=False):
            st.write(text)

        if "Final Answer:" in text:
            return text.split("Final Answer:", 1)[1].strip()

        action, action_input = None, None
        for line in text.splitlines():
            if ACTION_RE.match(line):
                action = ACTION_RE.match(line).group(1).title()
            if INPUT_RE.match(line):
                action_input = INPUT_RE.match(line).group(1).strip()

        if not action or not action_input:
            return "Could not understand next step."

        # Call selected tool
        if action == "Websearch":
            observation = tool_web_search(action_input)
        elif action == "Wikipedia":
            observation = tool_wikipedia(action_input)
        elif action == "Calculator":
            observation = tool_calculator(action_input)
        elif action == "Translator":
            observation = tool_translate(action_input)
        else:
            observation = f"Unknown tool: {action}"

        transcript.append(f"Thought: I will use {action}.")
        transcript.append(f"Action: {action}")
        transcript.append(f"Action Input: {action_input}")
        transcript.append(f"Observation: {observation}")

    summary = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Summarize briefly in English."},
            {"role": "user", "content": "\n".join(transcript)},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return summary.choices[0].message.content


# -----------------------------
# 5. STREAMLIT CHAT INTERFACE
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Type your question or message...")

if query:
    st.chat_message("user").write(query)
    st.session_state.chat_history.append(("user", query))

    if not api_key:
        st.error("Please provide your GROQ_API_KEY.")
    else:
        client = Groq(api_key=api_key)
        with st.spinner("Thinking..."):
            answer = ai_agent(client, model=model_name, user_input=query, max_iters=max_steps)
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append(("assistant", answer))
