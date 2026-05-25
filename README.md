# 🤖 LangChain Memory Voice Chatbot

A conversational AI chatbot built with LangChain and Gemma 4, featuring three memory types and full voice input/output support.

## Tech Stack

- **LangChain** — chain orchestration and memory management
- **Gemma 4 31B** — via Google AI Studio free API
- **Groq Whisper API** — speech to text (voice input)
- **gTTS** — text to speech (voice output)
- **Streamlit** — interactive chat UI

## Features

- 💬 Text and voice input
- 🔊 Voice output — bot speaks every response
- 🧠 Three switchable memory types
- 🎨 Clean chat bubble UI with message history

## Memory Types

### Buffer Memory — Remembers Everything
Stores the complete conversation history. The bot remembers every message from the start of the session.

### Window Memory — Remembers Last 5 Exchanges
Only keeps the last 10 messages (5 exchanges). Older messages are forgotten — useful for long conversations where recent context matters most.

### Summary Memory — Summarizes Old Messages
Compresses older messages into a brief summary using the LLM, then combines it with recent messages. Key facts persist even from very old messages without consuming too many tokens.

## Screenshots

### Buffer Memory — Full Conversation
[![Buffer Conversation](Screenshots/Buffer-conversation.png)](Screenshots/Buffer-conversation.png)

### Buffer Memory — Output
[![Buffer Output](Screenshots/Buffer-output.png)](Screenshots/Buffer-output.png)

### Summary Memory UI
[![Summary Memory UI](Screenshots/summery-memory-UI.png)](Screenshots/summery-memory-UI.png)

### Window Memory — UI
[![Window Memory UI](Screenshots/Window-memory-UI.png)](Screenshots/Window-memory-UI.png)

### Window Memory — Output (Forgetting Old Messages)
[![Window Memory Output](Screenshots/Window-memory-output.png)](Screenshots/Window-memory-output.png)

## How Memory Types Differ

| Memory Type | Remembers | Best For |
|-------------|-----------|----------|
| Buffer | Everything | Short conversations |
| Window (Last 5) | Last 5 exchanges only | Long conversations |
| Summary | Key points from all messages | Very long sessions |

## Note on Whisper

Local OpenAI Whisper was too heavy for Streamlit — caused connection timeouts. Switched to **Groq's Whisper API** (`whisper-large-v3`). Audio is recorded locally via `sounddevice`, converted to WAV in memory, and sent to Groq for transcription. Same model quality, runs on Groq's GPU instead of local CPU.

## How to Run

1. Clone the repository
```bash
git clone https://github.com/MohomedShajith/langchain-memory-voice-chatbot.git
cd langchain-memory-voice-chatbot
```

2. Create and activate virtual environment
```bash
python -m venv chat_env
chat_env\Scripts\activate
```

3. Install dependencies
```bash
pip install langchain langchain-google-genai langchain-community google-generativeai streamlit python-dotenv groq sounddevice scipy gtts
```

4. Create `.env` file
```env
GOOGLE_API_KEY=your_google_ai_studio_key
GROQ_API_KEY=your_groq_api_key
```

5. Run
```bash
streamlit run app.py
```

## Key Concepts Learned

- **LangChain modern approach (v0.3+)** — `MessagesPlaceholder` + `st.session_state` replaces deprecated `ConversationBufferMemory`
- **Pipe operator** — `prompt | llm | StrOutputParser()` connects components
- **Gemma 4 thinking blocks** — model returns reasoning + text, must extract text only
- **Memory as context** — chat history injected into every prompt via `MessagesPlaceholder`

## Related Projects

- [Patient Readmission Prediction](https://github.com/MohomedShajith/Patient-Readmission-Prediction)
- [Movie Recommendation System](https://github.com/MohomedShajith/movie-recommendation)
- [Financial Fraud Detection](https://github.com/MohomedShajith/Financial-Fraud-Detection-System)
- [Real Time Stock Dashboard](https://github.com/MohomedShajith/Real-Time-Stock-Dashboard)
