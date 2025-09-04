import streamlit as st
import librosa
import numpy as np

st.title("🎸 AI Guitar Helper")
st.write("Upload a song and I’ll guess the chords & strumming pattern!")

# Upload file
uploaded_file = st.file_uploader("Choose a song file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, duration=20)  # take first 20 sec
    
    # Extract harmonic part (for chords)
    y_harmonic, _ = librosa.effects.hpss(y)
    
    # Get chroma (pitches over time)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Simple chord mapping (very rough!)
    chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    main_chord = chords[np.argmax(chroma_mean)]
    
    # Beat tracking (for strumming pattern)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    st.subheader("Results:")
    st.write(f"**Main Chord Detected:** {main_chord}")
    st.write(f"**Tempo:** {int(tempo)} BPM")
    
    # Fake strumming suggestion
    if tempo < 90:
        strum = "D-DU-UDU"
    elif tempo < 130:
        strum = "DDU UDU"
    else:
        strum = "DUDUDU"
    
    st.write(f"**Suggested Strumming Pattern:** {strum}")
