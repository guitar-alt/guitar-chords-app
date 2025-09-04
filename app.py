import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.title("🎸 AI Guitar Helper")
st.write("Upload a song and I’ll guess the chords & strumming pattern!")

# Upload file
uploaded_file = st.file_uploader("Choose a song file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Load audio (first 20 sec for speed)
    y, sr = librosa.load(uploaded_file, duration=20)

    # Extract harmonic part (for chords)
    y_harmonic, _ = librosa.effects.hpss(y)

    # Get chroma (pitches over time)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

    # Chord names
    chords = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

    progression = []
    times = []
    window_size = int(3 * sr / 512)  # ~3 sec windows

    for i in range(0, chroma.shape[1], window_size):
        window = np.mean(chroma[:, i:i+window_size], axis=1)
        if window.size > 0:
            chord = chords[np.argmax(window)]
            progression.append(chord)
            times.append(i * 512 / sr)  # convert frame index to time in seconds

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
        ax.barh(0, width=3, left=times[i], height=0.5, align="center", label=chord if i == 0 else "")

        # Label chord above the bar
        ax.text(times[i] + 1.5, 0.1, chord, ha="center", va="bottom", fontsize=10, rotation=0)

    ax.set_xlim(0, max(times) + 3)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Chord Progression Timeline")
    st.pyplot(fig)
