import uuid
import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ShopNow Support",
    page_icon="🛍️",
    layout="centered"
)


# ── Helper defined FIRST so it can be called anywhere below ──────────────────
def render_response_metadata(msg: dict, session_id: str):
    confidence = msg.get("confidence", 0)
    escalated  = msg.get("escalated", False)
    sources    = msg.get("sources", [])

    # Confidence badge — colour coded
    col1, col2 = st.columns([3, 1])
    with col2:
        pct = int(confidence * 100)
        if confidence >= 0.6:
            st.success(f"✅ {pct}% confident")
        elif confidence >= 0.35:
            st.warning(f"⚠️ {pct}% confident")
        else:
            st.error(f"❌ {pct}% confident")

    # Escalation notice
    if escalated:
        st.info(
            "📋 **Your question has been flagged for human review.** "
            "A ShopNow support agent will follow up with you via email within 24 hours. "
            f"Please reference session ID `{session_id[:12]}` if you call us.",
            icon="📋"
        )

    # Sources expander
    if sources:
        with st.expander(f"📚 View sources ({len(sources)})"):
            for src in sources:
                st.markdown(f"**{src['file']}**")
                st.caption(src["excerpt"])
                st.divider()


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_error" not in st.session_state:
    st.session_state.api_error = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛍️ ShopNow")
    st.caption("Customer Support Chat")
    st.divider()

    st.markdown("**I can help you with:**")
    st.markdown(
        "- 📦 Shipping & delivery\n"
        "- 🔄 Returns & refunds\n"
        "- 💳 Payments & promotions\n"
        "- 🛒 Products & availability\n"
        "- 👤 Account & orders"
    )
    st.divider()

    st.caption("Your session reference:")
    st.code(st.session_state.session_id[:12], language=None)

    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.api_error = False
        st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("ShopNow Customer Support 🛍️")
st.caption("Hi! I'm ShopNow's virtual assistant. Ask me about orders, shipping, returns, or products.")

if st.session_state.api_error:
    st.error(
        "⚠️ Cannot reach the support server. Make sure the backend is running:\n"
        "```\nuvicorn backend.main:app --reload --port 8000\n```",
        icon="🔌"
    )

st.divider()

# ── Chat history (function is defined above so safe to call here) ─────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            render_response_metadata(msg, st.session_state.session_id)

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your order, shipping, or products..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking up your answer..."):
            try:
                resp = httpx.post(
                    f"{API_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt
                    },
                    timeout=60.0
                )
                resp.raise_for_status()
                data       = resp.json()
                answer     = data["answer"]
                confidence = data["confidence"]
                escalated  = data["escalated"]
                sources    = data.get("sources", [])
                st.session_state.api_error = False

            except Exception:
                answer     = "Sorry, I'm having trouble connecting. Please try again in a moment."
                confidence = 0.0
                escalated  = False
                sources    = []
                st.session_state.api_error = True

        st.write(answer)

        msg_data = {
            "role":       "assistant",
            "content":    answer,
            "confidence": confidence,
            "escalated":  escalated,
            "sources":    sources
        }
        render_response_metadata(msg_data, st.session_state.session_id)
        st.session_state.messages.append(msg_data)
