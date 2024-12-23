import streamlit as st
import sounddevice as sd
import wave
import torch
import os
import tempfile
import threading
import time
import asyncio
import edge_tts
import numpy as np
from typing import List, Optional
from faster_whisper import WhisperModel
import groq
import pygame

# Configure page
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_playing' not in st.session_state:
    st.session_state.audio_playing = False
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'response_queue' not in st.session_state:
    st.session_state.response_queue = []
if 'mic_initialized' not in st.session_state:
    st.session_state.mic_initialized = False

def initialize_audio():
    try:
        # Try different audio configurations
        try:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
        except pygame.error:
            try:
                pygame.mixer.init()  # Try default initialization
            except pygame.error:
                # If pygame fails, try without audio initialization
                st.warning("⚠️ Audio output might not work properly on this system.")
                pass
    except Exception as e:
        st.error(f"Audio initialization error: {str(e)}")

# Initialize audio
initialize_audio()

# Initialize clients with provided keys
@st.cache_resource
def initialize_clients():
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # Initialize Groq client
        groq_client = groq.Groq(
            api_key="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"
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        whisper_model = WhisperModel(
            model_size_or_path="base.en",
            device=device,
            compute_type="float16" if device == 'cuda' else "float32"
        )
        
        return groq_client, whisper_model
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None

groq_client, whisper_model = initialize_clients()

class OptimizedAudioPlayer:
    def __init__(self):
        self._stop_event = threading.Event()
        self.VOICE = "en-US-JennyNeural"
        self.audio_enabled = True
        
        # Check if audio is available
        try:
            pygame.mixer.get_init()
        except:
            self.audio_enabled = False
            st.warning("⚠️ Audio playback is not available on this system.")
    
    def stop(self):
        if self.audio_enabled:
            self._stop_event.set()
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    async def _generate_speech(self, text: str, output_file: str):
        communicate = edge_tts.Communicate(text, self.VOICE)
        await communicate.save(output_file)
    
    def play(self, text: str):
        if not text:
            return
            
        if not self.audio_enabled:
            st.warning("⚠️ Audio playback is not available. Displaying text only.")
            st.write(text)
            return
            
        self._stop_event.clear()
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
            
            # Generate speech using Edge TTS
            asyncio.run(self._generate_speech(text, temp_path))
            
            # Play using pygame if available
            try:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for playback to finish or stop event
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"Playback error: {str(e)}")
                st.write(text)  # Fallback to displaying text
            
            # Cleanup temporary file
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"🔇 Audio generation error: {str(e)}")
            st.write(text)  # Fallback to displaying text
        finally:
            st.session_state.audio_playing = False

# Remove OptimizedAudioRecorder class as we'll use st.audio_input instead

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



def process_response(text_input: str, is_voice: bool = False):
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
        
        st.session_state.audio_playing = True
        player_thread = threading.Thread(
            target=st.session_state.audio_player.play,
            args=(response,),
            daemon=True
        )
        player_thread.start()
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")

def play_stored_response(response_text: str):
    """Play a stored assistant response"""
    if st.session_state.audio_playing:
        st.session_state.audio_player.stop()
    
    st.session_state.audio_playing = True
    player_thread = threading.Thread(
        target=st.session_state.audio_player.play,
        args=(response_text,),
        daemon=True
    )
    player_thread.start()

# Initialize components
if 'audio_player' not in st.session_state:
    st.session_state.audio_player = OptimizedAudioPlayer()

if not st.session_state.mic_initialized:
    try:
        st.session_state.mic_initialized = st.session_state.audio_recorder.initialize_microphone()
    except Exception as e:
        st.error(f"Error initializing microphone: {str(e)}")

# Main UI
st.title("🤖 AI Voice Assistant")

# Add this before the chat container
if not st.session_state.mic_initialized:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.warning("🎤 Click the button to enable microphone access")
    with col2:
        if st.button("🎙️ Enable Microphone", use_container_width=True):
            if st.session_state.audio_recorder.initialize_microphone():
                st.balloons()
                st.success("✅ Microphone initialized successfully!")
                time.sleep(1)
                st.experimental_rerun()
            else:
                st.error("❌ Could not initialize microphone")
                st.info("""
                💡 Troubleshooting:
                1. Click the lock/camera icon in your browser's address bar
                2. Enable microphone access
                3. Refresh the page
                4. Try clicking the button again
                """)

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
                    play_stored_response(message["content"])

# Input area with audio input
col1, col2 = st.columns([6, 1])

with col1:
    text_input = st.text_input(
        "",
        placeholder="Type your message here...",
        key="text_input",
        label_visibility="collapsed"
    )

with col2:
    audio_bytes = st.audio_input("🎤", key="audio_input")
    if audio_bytes:
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            f.write(audio_bytes)
            audio_file = f.name
        
        with st.spinner("Processing..."):
            user_text = transcribe_audio(audio_file)
            if user_text:
                process_response(user_text, is_voice=True)
        os.unlink(audio_file)  # Clean up temp file
        st.experimental_rerun()

# Handle text input
if text_input and text_input != st.session_state.processed_text:
    st.session_state.processed_text = text_input
    process_response(text_input)
    st.experimental_rerun()

# Clear chat button
if st.button("🗑️ Clear Chat", key="clear", help="Clear all messages"):
    if st.session_state.audio_playing:
        st.session_state.audio_player.stop()
    st.session_state.history = []
    st.session_state.processed_text = None
    st.experimental_rerun()
