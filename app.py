"""
Jarvis - Personalized AI Assistant
Streamlit UI for document upload, chat, and voice interaction
"""

import streamlit as st
import logging
from pathlib import Path
from datetime import datetime
import time

from core.config import get_config
from core.user_manager import UserManager
from core.rag_engine import RAGEngine
from voice.asr_engine import ASREngine
from voice.tts_engine import TTSEngine
from utils.helpers import (
    setup_logging,
    is_supported_document,
    format_file_size,
    validate_user_id,
    sanitize_user_id
)

# Setup
setup_logging()
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Jarvis - Personal AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .source-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'config' not in st.session_state:
    st.session_state.config = get_config()


def init_user(user_id: str):
    """Initialize user and RAG engine"""
    try:
        st.session_state.user_id = user_id
        st.session_state.rag_engine = RAGEngine(user_id)
        st.session_state.chat_history = []
        st.success(f"✅ Initialized session for user: {user_id}")
    except Exception as e:
        st.error(f"❌ Error initializing user: {e}")
        logger.error(f"Error initializing user {user_id}: {e}")


def sidebar():
    """Render sidebar with user selection and stats"""
    with st.sidebar:
        st.title("🤖 Jarvis")
        st.markdown("*Your Personal AI Assistant*")
        st.divider()

        # User selection
        st.subheader("👤 User")

        user_manager = UserManager()
        existing_users = [u["user_id"] for u in user_manager.list_users()]

        tab1, tab2 = st.tabs(["Existing User", "New User"])

        with tab1:
            if existing_users:
                selected_user = st.selectbox(
                    "Select user",
                    options=existing_users,
                    key="existing_user_select"
                )
                if st.button("Load User", key="load_existing"):
                    init_user(selected_user)
            else:
                st.info("No users yet. Create a new user!")

        with tab2:
            new_user = st.text_input(
                "User ID",
                placeholder="Enter user ID (e.g., john_doe)",
                key="new_user_input"
            )
            if st.button("Create User", key="create_new"):
                is_valid, error = validate_user_id(new_user)
                if is_valid:
                    sanitized = sanitize_user_id(new_user)
                    init_user(sanitized)
                else:
                    st.error(f"Invalid user ID: {error}")

        st.divider()

        # Show stats if user is logged in
        if st.session_state.user_id:
            st.subheader("📊 Stats")
            stats = st.session_state.rag_engine.get_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats["document_count"])
                st.metric("Index Size", f"{stats['index_size_mb']:.1f} MB")
            with col2:
                st.metric(
                    "Savings",
                    f"{stats['storage_savings_percent']:.0f}%"
                )
                st.metric(
                    "Index",
                    "✅" if stats["index_built"] else "❌"
                )

        st.divider()

        # Configuration info
        with st.expander("⚙️ Configuration"):
            config = st.session_state.config
            st.write(f"**LLM:** {config.llm.provider}")
            st.write(f"**Embedding:** {config.leann.embedding_model}")
            st.write(f"**Web Search:** {'✅' if config.web_search.enabled else '❌'}")
            st.write(f"**ASR:** {'✅' if config.asr.enabled else '❌'}")
            st.write(f"**TTS:** {'✅' if config.tts.enabled else '❌'}")


def document_management_tab():
    """Document upload and management"""
    st.header("📄 Document Management")

    if not st.session_state.user_id:
        st.warning("⚠️ Please select or create a user first")
        return

    user_manager = UserManager()
    rag_engine = st.session_state.rag_engine

    # Upload section
    st.subheader("Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, MD, and more"
    )

    if uploaded_files:
        if st.button("💾 Save Documents"):
            with st.spinner("Saving documents..."):
                saved_count = 0
                for uploaded_file in uploaded_files:
                    if is_supported_document(uploaded_file.name):
                        try:
                            user_manager.save_document(
                                st.session_state.user_id,
                                uploaded_file.name,
                                uploaded_file.read()
                            )
                            saved_count += 1
                        except Exception as e:
                            st.error(f"Error saving {uploaded_file.name}: {e}")
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.name}")

                if saved_count > 0:
                    st.success(f"✅ Saved {saved_count} document(s)")
                    st.rerun()

    st.divider()

    # Document list
    st.subheader("Your Documents")

    documents = user_manager.get_user_documents(st.session_state.user_id)

    if documents:
        for doc in documents:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.text(f"📄 {doc.name}")
            with col2:
                st.text(format_file_size(doc.stat().st_size))
            with col3:
                if st.button("🗑️", key=f"delete_{doc.name}"):
                    user_manager.delete_document(
                        st.session_state.user_id,
                        doc.name
                    )
                    st.success(f"Deleted {doc.name}")
                    st.rerun()
    else:
        st.info("No documents yet. Upload some to get started!")

    st.divider()

    # Index management
    st.subheader("Index Management")

    stats = rag_engine.get_stats()

    if stats["index_built"]:
        st.success("✅ Index is up to date")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Index"):
                with st.spinner("Rebuilding index..."):
                    result = rag_engine.build_index(force_rebuild=True)
                    if result["status"] == "success":
                        st.success(f"✅ {result['message']}")
                        st.info(f"Chunks: {result['chunk_count']}, Savings: {result['storage_savings_percent']:.0f}%")
                    else:
                        st.error(f"❌ {result['message']}")
        with col2:
            if st.button("🗑️ Delete Index"):
                user_manager.delete_index(st.session_state.user_id)
                st.success("Index deleted")
                st.rerun()
    else:
        if documents:
            st.warning("⚠️ Index needs to be built")
            if st.button("🔨 Build Index"):
                with st.spinner("Building index... This may take a few minutes."):
                    result = rag_engine.build_index()
                    if result["status"] == "success":
                        st.success(f"✅ {result['message']}")
                        st.balloons()
                        st.info(f"Chunks: {result['chunk_count']}, Savings: {result['storage_savings_percent']:.0f}%")
                        st.rerun()
                    else:
                        st.error(f"❌ {result['message']}")
        else:
            st.info("Upload documents first to build an index")


def chat_tab():
    """Chat interface"""
    st.header("💬 Chat with Jarvis")

    if not st.session_state.user_id:
        st.warning("⚠️ Please select or create a user first")
        return

    rag_engine = st.session_state.rag_engine
    stats = rag_engine.get_stats()

    if not stats["index_built"]:
        st.warning("⚠️ Build an index first to chat with your documents")
        if st.button("Go to Documents"):
            st.switch_page("app.py")
        return

    # Chat settings
    with st.expander("⚙️ Chat Settings"):
        col1, col2 = st.columns(2)
        with col1:
            use_web_fallback = st.checkbox(
                "Enable web search fallback",
                value=True,
                help="Automatically search the web if documents don't have the answer"
            )
        with col2:
            stream_response = st.checkbox(
                "Stream responses",
                value=True,
                help="Show response as it's being generated"
            )

    st.divider()

    # Chat history display
    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"📚 Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"], 1):
                            if source["type"] == "document":
                                st.markdown(f"**{i}. 📄 {source['title']}**")
                                st.caption(f"Relevance: {source['score']:.2f}")
                            else:  # web
                                st.markdown(f"**{i}. 🌐 {source['title']}**")
                                st.caption(source['url'])
                            st.text(source["preview"])
                            st.divider()

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            sources_placeholder = st.empty()

            try:
                if stream_response:
                    # Streaming response
                    full_response = ""
                    sources = []
                    used_web = False

                    for chunk in rag_engine.ask(
                        prompt,
                        use_web_fallback=use_web_fallback,
                        stream=True
                    ):
                        if chunk["type"] == "metadata":
                            sources = chunk["sources"]
                            used_web = chunk["used_web_search"]
                        elif chunk["type"] == "chunk":
                            full_response += chunk["content"]
                            message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)

                    # Show sources
                    if sources:
                        with sources_placeholder.expander(f"📚 Sources ({len(sources)})"):
                            for i, source in enumerate(sources, 1):
                                if source["type"] == "document":
                                    st.markdown(f"**{i}. 📄 {source['title']}**")
                                    st.caption(f"Relevance: {source['score']:.2f}")
                                else:
                                    st.markdown(f"**{i}. 🌐 {source['title']}**")
                                    st.caption(source['url'])
                                st.text(source["preview"])
                                st.divider()

                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": sources
                    })

                else:
                    # Non-streaming response
                    with st.spinner("Thinking..."):
                        response = rag_engine.ask(
                            prompt,
                            use_web_fallback=use_web_fallback,
                            stream=False
                        )

                    message_placeholder.markdown(response["answer"])

                    # Show sources
                    if response["sources"]:
                        with sources_placeholder.expander(f"📚 Sources ({len(response['sources'])})"):
                            for i, source in enumerate(response["sources"], 1):
                                if source["type"] == "document":
                                    st.markdown(f"**{i}. 📄 {source['title']}**")
                                    st.caption(f"Relevance: {source['score']:.2f}")
                                else:
                                    st.markdown(f"**{i}. 🌐 {source['title']}**")
                                    st.caption(source['url'])
                                st.text(source["preview"])
                                st.divider()

                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })

            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Chat error: {e}")

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


def main():
    """Main app"""
    # Render sidebar
    sidebar()

    # Main content
    if not st.session_state.user_id:
        st.title("🤖 Welcome to Jarvis")
        st.markdown("""
        ### Your Personal AI Assistant

        **Features:**
        - 📄 Upload and index your documents
        - 💬 Chat with your documents using natural language
        - 🌐 Automatic web search fallback
        - 🗣️ Multilingual voice support (coming soon)
        - 🔒 100% privacy - everything runs locally

        **Get Started:**
        1. Select an existing user or create a new one in the sidebar →
        2. Upload your documents
        3. Build an index
        4. Start chatting!

        """)

        # Configuration warnings
        issues = st.session_state.config.validate()
        if issues:
            with st.expander("⚠️ Configuration Warnings"):
                for issue in issues:
                    st.warning(issue)

    else:
        # Tabs for different features
        tab1, tab2 = st.tabs(["📄 Documents", "💬 Chat"])

        with tab1:
            document_management_tab()

        with tab2:
            chat_tab()


if __name__ == "__main__":
    main()
