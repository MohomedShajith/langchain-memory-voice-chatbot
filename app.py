import os
import io
import numpy as np
import streamlit as st
import scipy.io.wavfile as wav
import sounddevice
from gtts import gTTS
from groq import Groq
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.title("🤖 LangChain Memory Chatbot")

st.sidebar.title("⚙️ Settings")
memory_type = st.sidebar.selectbox(
    "Memory Type",
    ["Buffer", "Window (Last 5)", "Summary"]
)

llm = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain = prompt | llm

def record_and_transcribe():
    seconds = 5
    samplerate = 16000
    audio = sounddevice.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sounddevice.wait()
    audio = (audio.flatten() * 32768).astype(np.int16)
    buffer = io.BytesIO()
    wav.write(buffer, samplerate, audio)
    buffer.seek(0)
    buffer.name = "audio.wav"
    transcription = groq_client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=buffer
    )
    return transcription.text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)

# Input area
col1, col2 = st.columns([6, 1])
with col1:
    user_input = st.chat_input("Type your message...")
with col2:
    voice_clicked = st.button("🎤")

# Handle voice input
if voice_clicked:
    with st.spinner("Recording for 5 seconds... Speak now!"):
        user_input = record_and_transcribe()
    st.success(f"You said: {user_input}")

# Process input
if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if memory_type == "Buffer":
                active_history = st.session_state.chat_history
            elif memory_type == "Window (Last 5)":
                active_history = st.session_state.chat_history[-10:]
            elif memory_type == "Summary":
                if len(st.session_state.chat_history) > 4:
                    old_messages = st.session_state.chat_history[:-4]
                    recent_messages = st.session_state.chat_history[-4:]
                    summary_response = llm.invoke(
                        f"Summarize this conversation briefly: {old_messages}"
                    )
                    summary_text = summary_response.content if isinstance(summary_response.content, str) else summary_response.content[1]['text']
                    active_history = [AIMessage(content=f"Previous conversation summary: {summary_text}")] + recent_messages
                else:
                    active_history = st.session_state.chat_history

            response = chain.invoke({
                "input": user_input,
                "chat_history": active_history
            })

            bot_text = ""
            if isinstance(response.content, list):
                for block in response.content:
                    if block.get("type") == "text":
                        bot_text = block['text']
                        break
            else:
                bot_text = response.content

            st.write(bot_text)

            # Text to speech
            audio_buffer = text_to_speech(bot_text)
            st.audio(audio_buffer, format="audio/mp3")

    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=bot_text))