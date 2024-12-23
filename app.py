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

class OptimizedAudioRecorder:
    def __init__(self):
        self.sample_rate = 16000
        self.channels = 1
        self.dtype = np.int16
        self.recording = False
        self.audio_data = []
        self.stream = None
        self.device_id = None
    
    def initialize_microphone(self):
        try:
            # Add explicit HTML5 audio element to force permission prompt
            st.markdown("""
                <div>
                    <audio id="audio" style="display:none"></audio>
                    <script>
                        async function requestMicrophonePermission() {
                            try {
                                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                                document.getElementById('audio').srcObject = stream;
                            } catch(err) {
                                console.error('Microphone permission denied:', err);
                            }
                        }
                        requestMicrophonePermission();
                    </script>
                </div>
            """, unsafe_allow_html=True)
            
            # Give browser time to show permission prompt
            with st.spinner("🎤 Waiting for microphone permission..."):
                time.sleep(2)
            
            # Test microphone access
            test_stream = sd.InputStream(samplerate=16000, channels=1)
            test_stream.start()
            dummy_data = test_stream.read(1000)
            test_stream.stop()
            test_stream.close()
            
            # Get available devices
            devices = sd.query_devices()
            input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            
            if input_devices:
                self.device_id = input_devices[0]
                device_info = sd.query_devices(self.device_id)
                st.session_state.mic_initialized = True
                return True
            return False
        except Exception as e:
            st.error("❌ Microphone access denied or not available")
            st.error(f"Error details: {str(e)}")
            st.markdown("""
                ### 🔧 Troubleshooting Steps:
                1. Look for the microphone icon in your browser's address bar
                2. Click it and select "Allow"
                3. If you don't see the icon, click the lock/info icon
                4. Find "Microphone" in site settings and set to "Allow"
                5. Refresh the page and try again
            """)
            return False
    
    def start_recording(self):
        if self.device_id is None:
            st.error("❌ No microphone available. Please check permissions and try again.")
            return
        
        try:
            self.recording = True
            self.audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f'Audio callback status: {status}')
                if self.recording:
                    self.audio_data.append(indata.copy())
            
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=self.dtype,
                callback=callback,
                blocksize=1024,
                latency='low'
            )
            self.stream.start()
            st.info("🎤 Recording started...")
        except Exception as e:
            st.error(f"❌ Recording error: {str(e)}")
            self.recording = False

    def stop_recording(self):
        if not self.recording:
            return None
            
        try:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            
            if not self.audio_data:
                return None
                
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # Save to WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                with wave.open(f.name, 'wb') as wf:
                    wf.setnchannels(self.channels)
                    wf.setsampwidth(2)  # 16-bit audio
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(audio_data.tobytes())
                return f.name
        except Exception as e:
            st.error(f"Error stopping recording: {str(e)}")
            return None

    # Remove the old callback and __del__ methods as they're not needed
    def callback(self, in_data, frame_count, time_info, status):
        pass  # Remove this method

    def __del__(self):
        pass  # Remove this method

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

# Initialize components
if 'audio_recorder' not in st.session_state:
    st.session_state.audio_recorder = OptimizedAudioRecorder()
if 'audio_player' not in st.session_state:
    st.session_state.audio_player = OptimizedAudioPlayer()

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
    for message in st.session_state.history:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(
            f"""<div class="{role_class}">
                <strong>{'You' if message["role"] == "user" else '🤖 Assistant'}</strong>: {message["content"]}
                </div>""",
            unsafe_allow_html=True
        )

# Input area
col1, col2 = st.columns([6, 1])

with col1:
    text_input = st.text_input(
        "",
        placeholder="Type your message here...",
        key="text_input",
        label_visibility="collapsed"
    )

with col2:
    record_button_class = "record-button recording" if st.session_state.recording else "record-button"
    record_icon = "⏺️" if not st.session_state.recording else "⏹️"
    
    if st.button(record_icon, key="record_button", disabled=not st.session_state.mic_initialized):
        if not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.audio_recorder.start_recording()
        else:
            st.session_state.recording = False
            with st.spinner("Processing..."):
                audio_file = st.session_state.audio_recorder.stop_recording()
                if audio_file:
                    user_text = transcribe_audio(audio_file)
                    if user_text:
                        process_response(user_text, is_voice=True)
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
