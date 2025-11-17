# 🤖 JARVIS - Personalized AI Assistant

Your personal AI assistant with document understanding, web search, and multilingual voice interaction.

## ✨ Features

- 📄 **Document RAG**: Upload any documents (PDF, DOCX, TXT, etc.) and ask questions
- 🌐 **Web Search Fallback**: Automatically searches the web when docs don't have the answer
- 🗣️ **Multilingual Voice**: Supports 1600+ languages for speech input/output
- 🔒 **Privacy-First**: All processing runs locally on your GPU
- 💾 **Efficient Storage**: 97% less storage than traditional RAG systems (thanks to LEANN)
- 👥 **Multi-User**: Separate knowledge base for each user

## 🛠️ Tech Stack

- **RAG**: LEANN + facebook/contriever embeddings
- **LLM**: DeepSeek API (configurable: OpenAI, Ollama)
- **Web Search**: Brave Search MCP
- **ASR**: Omnilingual CTC_1B (1600+ languages)
- **TTS**: XTTS-v2 (17 languages, voice cloning)
- **UI**: Streamlit

## 🚀 Quick Start

### 1. Installation

**Option A: Using setup script (recommended - auto-installs uv)**
```bash
chmod +x setup.sh
./setup.sh
```

**Option B: Manual installation with uv (faster)**
```bash
# Install uv (ultra-fast package installer)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (much faster than pip)
uv pip install -r requirements.txt
```

**Option C: Traditional pip method**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your API keys:
# - DEEPSEEK_API_KEY (required for LLM)
# - BRAVE_API_KEY (required for web search)
```

### 3. (Optional) Install Voice Features

```bash
# If you want voice support (ASR + TTS), run:
chmod +x install_voice.sh
./install_voice.sh

# This installs both ASR and TTS with numpy 1.26.4 to avoid conflicts
```

### 4. Run

```bash
# Start Streamlit UI
streamlit run app.py
```

## 📁 Project Structure

```
jarvis/
├── app.py                      # Streamlit UI
├── core/
│   ├── config.py              # Configuration loader
│   ├── user_manager.py        # User index management
│   ├── rag_engine.py          # LEANN + web search integration
│   ├── llm_handler.py         # LLM API wrapper
│   └── mcp_search.py          # Brave Search MCP
├── voice/
│   ├── asr_engine.py          # Omnilingual ASR
│   └── tts_engine.py          # XTTS-v2 TTS
├── utils/
│   └── helpers.py             # Utility functions
└── users/                      # Per-user data (auto-created)
    └── {user_id}/
        ├── documents/         # Uploaded files
        └── index.leann/       # LEANN index
```

## 🎯 Usage

### Text Chat (Default)

1. Enter your user ID
2. Upload documents
3. Ask questions in the chat

### Voice Chat (Optional)

1. Enable in `.env`:
   ```bash
   ASR_ENABLED=true
   TTS_ENABLED=true
   ```
2. Click microphone icon to speak
3. Hear AI responses

## 🌍 Multilingual Support

- **ASR**: 1600+ languages (Omnilingual)
- **TTS**: 17 languages (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja, hu, ko, hi)
- **LLM**: 100+ languages (DeepSeek)

## ⚙️ Configuration Options

See `.env.example` for all available settings.

### Key Settings:

- `SIMILARITY_THRESHOLD`: Controls when to fallback to web search (0.0-1.0)
- `TOP_K`: Number of document chunks to retrieve
- `WEB_SEARCH_ENABLED`: Enable/disable web fallback
- `ASR_ENABLED`/`TTS_ENABLED`: Enable voice features

## 💾 Storage Efficiency

Example with 100 PDFs (500MB):
- Traditional Vector DB: ~2GB index
- LEANN: ~60MB index (97% savings!)

## 🔧 Advanced Usage

### Using Different LLMs

**DeepSeek (default):**
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key
```

**OpenAI:**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o-mini
```

**Ollama (local):**
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

### Custom Voice

Clone your voice with XTTS-v2:
```bash
TTS_SPEAKER_WAV=/path/to/your/voice_sample.wav
```

## 🤝 Contributing

This is a personal project. Feel free to fork and customize!

## 📄 License

MIT License

## 🙏 Acknowledgments

- [LEANN](https://github.com/yichuan-w/LEANN) - Efficient vector indexing
- [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr) - Multilingual speech recognition
- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS-v2 text-to-speech

---

Built with ❤️ for personalized AI
