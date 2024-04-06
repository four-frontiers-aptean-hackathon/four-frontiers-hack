import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
from feature_output import get_features, get_title
from langchain.memory import ConversationBufferMemory

class VoiceRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self, filename, duration):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        self.stream = self.audio.open(format=FORMAT, channels=CHANNELS,
                                      rate=RATE, input=True,
                                      frames_per_buffer=CHUNK)
        st.write("Recording...")

        for _ in range(int(RATE / CHUNK * duration)):
            if self.is_recording:
                data = self.stream.read(CHUNK)
                self.frames.append(data)
            else:
                break

        st.write("Finished recording.")

        self.stream.stop_stream()
        self.stream.close()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

    def stop_recording(self):
        self.is_recording = False

def transcribe_audio(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text

def main():
    st.title("Welcome to the Requirements Whisperer!")
    st.subheader("Where ideas for the application become clear.")
    recorder = VoiceRecorder()
    duration = 15  # Set fixed duration for recording in seconds

    memory=ConversationBufferMemory()
    # Add a speech icon
    st.write("Choose input method:")
    speech_icon = "🎤"
    text_icon = "✍️"
    input_method = st.radio("", [speech_icon, text_icon])
    user_input=""
    if input_method == speech_icon:
        if st.button("Start Recording"):
            recorder.is_recording = True
            recorder.start_recording("recorded_audio.wav", duration)
        if st.button("Submit Speech"):
            recorder.stop_recording()
            st.write("Recording stopped.")
            st.write("Transcribing speech...")
            user_input = transcribe_audio("recorded_audio.wav")
            st.write("Transcribed Speech:")
            st.write(user_input)
    else:
        user_input = st.text_input("How can I help?")
    
    if st.button("Submit"):
        output=get_features(user_input,memory)
        st.write(output)
    
    for msg in memory.chat_memory.messages:
        print(msg.name)


if __name__ == "__main__":
    main()
