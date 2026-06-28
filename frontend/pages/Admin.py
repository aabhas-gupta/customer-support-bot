import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ShopNow Admin",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ ShopNow Admin Panel")
st.caption("Manage escalations, view conversations, and upload documents.")
st.divider()

tab_escalations, tab_conversations, tab_upload = st.tabs([
    "🚨 Escalations", "💬 Conversations", "📄 Upload Document"
])


# ── Tab 1: Escalations ────────────────────────────────────────────────────────
with tab_escalations:
    st.subheader("Open Escalations")
    st.caption("These are conversations flagged for human review.")

    show_resolved = st.toggle("Show resolved escalations", value=False)

    try:
        resp = httpx.get(
            f"{API_URL}/admin/escalations",
            params={"resolved": str(show_resolved).lower()},
            timeout=10.0
        )
        resp.raise_for_status()
        escalations = resp.json()
    except Exception:
        st.error("Could not reach the backend. Make sure it is running on port 8000.")
        escalations = []

    if not escalations:
        st.info("No escalations to show." if show_resolved else "No open escalations.")
    else:
        for esc in escalations:
            reason_label = {
                "no_answer": "No answer found",
                "low_confidence": "Low confidence answer"
            }.get(esc["reason"], esc["reason"])

            status = "Resolved" if esc["resolved"] else "Open"
            color  = "green"   if esc["resolved"] else "red"

            with st.expander(
                f":{color}[{status}] — Session `{esc['conversation_id'][:12]}` · {reason_label} · {esc['created_at'][:10]}"
            ):
                st.markdown(f"**Escalation ID:** {esc['id']}")
                st.markdown(f"**Reason:** {reason_label}")
                st.markdown(f"**Created:** {esc['created_at']}")

                if esc["resolved"]:
                    st.success(f"Resolved at {esc['resolved_at']}")
                    if esc["admin_notes"]:
                        st.markdown(f"**Admin notes:** {esc['admin_notes']}")
                else:
                    notes = st.text_area(
                        "Resolution notes (optional)",
                        key=f"notes_{esc['id']}",
                        placeholder="Describe how this was handled..."
                    )
                    if st.button("Mark as resolved", key=f"resolve_{esc['id']}"):
                        try:
                            r = httpx.post(
                                f"{API_URL}/admin/resolve/{esc['id']}",
                                json={"admin_notes": notes or None},
                                timeout=10.0
                            )
                            r.raise_for_status()
                            st.success("Escalation resolved.")
                            st.rerun()
                        except Exception:
                            st.error("Failed to resolve escalation.")


# ── Tab 2: Conversations ──────────────────────────────────────────────────────
with tab_conversations:
    st.subheader("All Conversations")

    try:
        resp = httpx.get(f"{API_URL}/admin/conversations", timeout=10.0)
        resp.raise_for_status()
        conversations = resp.json()
    except Exception:
        st.error("Could not reach the backend.")
        conversations = []

    if not conversations:
        st.info("No conversations yet.")
    else:
        st.caption(f"{len(conversations)} total conversation(s)")

        for conv in conversations:
            badges = []
            if conv["is_escalated"]:
                badges.append("🚨 Escalated")
            if conv["resolved"]:
                badges.append("✅ Resolved")
            badge_str = "  ·  ".join(badges) if badges else "Normal"

            label = f"Session `{conv['id'][:12]}` · {conv['created_at'][:10]} · {badge_str}"

            with st.expander(label):
                messages = conv.get("messages", [])
                if not messages:
                    st.caption("No messages.")
                else:
                    for msg in messages:
                        role = msg["role"]
                        icon = "🧑" if role == "user" else "🤖"
                        st.markdown(f"{icon} **{role.capitalize()}:** {msg['content']}")
                        if role == "assistant" and msg.get("confidence") is not None:
                            pct = int(msg["confidence"] * 100)
                            if msg["confidence"] >= 0.6:
                                st.caption(f"✅ {pct}% confident")
                            elif msg["confidence"] >= 0.35:
                                st.caption(f"⚠️ {pct}% confident")
                            else:
                                st.caption(f"❌ {pct}% confident")
                        st.divider()


# ── Tab 3: Upload Document ────────────────────────────────────────────────────
with tab_upload:
    st.subheader("Upload Document to Knowledge Base")
    st.caption("Accepted formats: `.txt`, `.pdf`, `.docx`, `.csv`")

    uploaded = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf", "docx", "csv"],
        help="The document will be chunked, embedded, and added to ChromaDB."
    )

    if uploaded:
        st.info(f"Ready to upload: **{uploaded.name}** ({uploaded.size:,} bytes)")

        if st.button("Upload and index", type="primary"):
            with st.spinner(f"Indexing {uploaded.name}..."):
                try:
                    resp = httpx.post(
                        f"{API_URL}/admin/upload",
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                        timeout=60.0
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(result["message"])
                    st.metric("Chunks added", result["chunks_added"])
                except httpx.HTTPStatusError as e:
                    st.error(f"Upload failed: {e.response.json().get('detail', str(e))}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")
