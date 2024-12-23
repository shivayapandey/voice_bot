import streamlit as st
from st_audiorec import st_audiorec
import torch
from typing import List
import os
import tempfile
import edge_tts
import asyncio
from faster_whisper import WhisperModel
import groq
import soundfile as sf

try:
    if st._is_running_with_streamlit:
        # Only run initialization when in Streamlit environment
        st.set_page_config(
            page_title="AI Voice Assistant",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
except:
    pass

# Styling
st.markdown("""
    <style>
    /* Dark theme with white text */
    :root {
        --primary-color: #4C61F0;
        --background-color: #000000;
        --text-color: #FFFFFF;
    }
    
    .stApp {
        background-color: #000000;
        color: var(--text-color);
    }
    
    /* Chat messages */
    .user-message {
        background: #1E1E1E;
        color: var(--text-color);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(255,255,255,0.05);
        border-left: 4px solid #4C61F0;
    }
    
    .assistant-message {
        background: #2D2D2D;
        color: var(--text-color);
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(255,255,255,0.05);
        border-left: 4px solid #88C0D0;
    }
    
    /* Input area styling */
    .stTextInput > div > div > input {
        background-color: #1E1E1E !important;
        color: var(--text-color) !important;
        border-color: #333333 !important;
    }

    /* Override Streamlit's default white background */
    .stTextInput, .stButton, .stMarkdown {
        color: var(--text-color) !important;
    }

    .st-emotion-cache-18ni7ap {
        background-color: #000000;
    }

    .st-emotion-cache-1y4p8pa {
        background-color: #000000;
    }

    /* Ensure any Streamlit elements maintain dark theme */
    .st-bb {
        background-color: #000000;
    }

    .st-emotion-cache-1wbqy5l {
        color: var(--text-color);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None

@st.cache_resource(show_spinner=False)
def initialize_clients():
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        torch.set_num_threads(1)  # Prevent threading issues
        
        groq_client = groq.Groq(
            api_key="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"
        )
        
        device = 'cpu'
        try:
            whisper_model = WhisperModel(
                model_size_or_path="base.en",
                device=device,
                compute_type="float32",
                num_workers=1  # Prevent threading issues
            )
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")
            return None, None
            
        return groq_client, whisper_model
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None

groq_client, whisper_model = initialize_clients()

async def generate_speech(text: str, output_file: str, voice="en-US-JennyNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

def play_audio(text: str):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_path = temp_file.name
            
        # Generate speech with timeout
        try:
            asyncio.run(asyncio.wait_for(generate_speech(text, temp_path), timeout=10.0))
        except asyncio.TimeoutError:
            st.error("Speech generation timed out")
            st.write(text)
            return
            
        # Play using streamlit's native audio player
        try:
            with open(temp_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
        except Exception as e:
            st.error(f"Audio playback error: {str(e)}")
            st.write(text)
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        st.error(f"Audio error: {str(e)}")
        st.write(text)

@st.cache_data(ttl=300)
def transcribe_audio(audio_file: str) -> str:
    if not audio_file or not whisper_model:
        return ""
        
    try:
        segments, _ = whisper_model.transcribe(
            audio_file,
            beam_size=1,
            word_timestamps=False,
            language='en',
            vad_filter=True
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def get_assistant_response(messages: List[dict]) -> str:
    if not groq_client:
        return "Error: Groq client not properly initialized"
        
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return f"Error: {str(e)}"

def process_response(text_input: str):
    if not text_input:
        return
        
    st.session_state.history.append({"role": "user", "content": text_input})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise."}
    ] + st.session_state.history[-6:]
    
    try:
        response = get_assistant_response(messages)
        if response.startswith("Error:"):
            st.error(response)
            return
            
        st.session_state.history.append({"role": "assistant", "content": response})
        play_audio(response)
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")

# Main UI
st.title("🤖 AI Voice Assistant")

# Chat container
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.history):
        col1, col2 = st.columns([6, 1])
        with col1:
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(
                f"""<div class="{role_class}">
                    <strong>{'You' if message["role"] == "user" else '🤖 Assistant'}</strong>: {message["content"]}
                    </div>""",
                unsafe_allow_html=True
            )
        with col2:
            if message["role"] == "assistant":
                if st.button("🔊", key=f"play_{i}", help="Play response"):
                    play_audio(message["content"])

# Input area
col1, col2 = st.columns([5, 1])

with col1:
    text_input = st.text_input("", placeholder="Type your message here...", key="text_input", label_visibility="collapsed")

with col2:
    wav_audio_data = st_audiorec()
    if wav_audio_data is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(wav_audio_data)
            audio_file = f.name
            
        try:
            with st.spinner("Processing audio..."):
                user_text = transcribe_audio(audio_file)
                if user_text:
                    process_response(user_text)
                os.unlink(audio_file)
            st.rerun()
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")

if text_input and text_input != st.session_state.processed_text:
    st.session_state.processed_text = text_input
    process_response(text_input)
    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat", key="clear"):
    st.session_state.history = []
    st.session_state.processed_text = None
    st.rerun()
