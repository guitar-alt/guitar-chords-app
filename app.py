import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.title("🎸 AI Guitar Helper")
st.write("Upload a song and I’ll guess the chords & strumming pattern!")

# Upload file
uploaded_file = st.file_uploader("Choose a song file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Get total duration of the file
    total_duration = librosa.get_duration(path=uploaded_file)

    # Chord names
    chords = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

    progression = []
    times = []
    chunk_size = 15  # seconds per chunk (faster!)

    for start in range(0, int(total_duration), chunk_size):
        # Load only a small part (chunk) of the song
        y, sr = librosa.load(uploaded_file, offset=start, duration=chunk_size, sr=11025)  # downsample for speed

        # Extract harmonic part (for chords)
        y_harmonic, _ = librosa.effects.hpss(y)

        # Get chroma
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)  # faster than chroma_cqt
        chroma_mean = np.mean(chroma, axis=1)

        chord = chords[np.argmax(chroma_mean)]
        progression.append(chord)
        times.append(start)

    # Beat tracking (for strumming pattern)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # Display results
    st.subheader("Results:")
    st.write("**Chord Progression (approx):**")
    st.write(" → ".join(progression))

    st.write(f"**Tempo:** {int(tempo)} BPM")

    # Strumming suggestion
    if tempo < 90:
        strum = "D-DU-UDU"
    elif tempo < 130:
        strum = "DDU UDU"
    else:
        strum = "DUDUDU"

    st.write(f"**Suggested Strumming Pattern:** {strum}")

    # --- Timeline Visualization ---
    st.subheader("Chord Timeline")
    fig, ax = plt.subplots(figsize=(10, 2))

    for i, chord in enumerate(progression):
        ax.barh(0, width=chunk_size, left=times[i], height=0.5,
                align="center", label=chord if i == 0 else "")
        ax.text(times[i] + chunk_size/2, 0.1, chord, ha="center", va="bottom", fontsize=10)

    ax.set_xlim(0, max(times) + chunk_size)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Chord Progression Timeline")
    st.pyplot(fig)

   
