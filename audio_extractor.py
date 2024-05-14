import streamlit as st
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from PIL import Image


def calculate_jitter(signal, sampling_rate):
    # Calculate fundamental frequency (F0)
    f0 = librosa.yin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    
    # Calculate period of each pitch period
    periods = 1. / f0
    
    # Calculate jitter
    jitter_abs = np.mean(np.abs(np.diff(periods))) 
    jitter_percent= jitter_abs / np.mean(periods) 
    
    return jitter_percent, jitter_abs

def calculate_shimmer(signal):
    # Calculate amplitude envelope
    amplitude_envelope = np.abs(librosa.amplitude_to_db(librosa.amplitude_to_db(signal)))
    
    # Calculate shimmer
    shimmer = np.mean(np.abs(np.diff(amplitude_envelope))) / np.mean(amplitude_envelope)
    shimmer_db = 20 * np.log10(shimmer)

    # Calculate APQ3
    apq3 = np.mean(np.abs(np.diff(amplitude_envelope, n=2))) / np.mean(amplitude_envelope)

    # Calculate APQ5
    apq5 = np.mean(np.abs(np.diff(amplitude_envelope, n=4))) / np.mean(amplitude_envelope)

    # Calculate MDVP:APQ
    mdvp_apq = np.mean(np.abs(np.diff(amplitude_envelope))) / np.mean(amplitude_envelope)

    # Calculate DDA
    dda = apq3 * 3

    return shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda


def main():
    st.title("Voice Analysis Tool")
    
    st.info("Preferably use audio input of max 1min as this is used for audio processing. Wait for few mins and the output will be generated.")
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac'])

    if uploaded_file is not None:

        st.write("File Uploaded Successfully!")
        
        st.write("Processing the audio file....")

        signal, sampling_rate = librosa.load(uploaded_file, sr=None)

        # Calculate features
        f0, _, _ = librosa.pyin(signal, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0 = f0[~np.isnan(f0)]  # remove nan values

        plt.figure(figsize=(14, 5))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.savefig('plot2.png', bbox_inches='tight')
        # plt.show()

        image2 = Image.open('plot2.png')

        st.image(image2, caption='Audio_spectogram', use_column_width=True)

        
        jitter_percent, jitter_abs = calculate_jitter(signal, sampling_rate)
        shimmer, shimmer_db, apq3, apq5, mdvp_apq, dda = calculate_shimmer(signal)

        harmonic, percussive = librosa.effects.hpss(signal)
        
        # Create DataFrame
        features_dict = {
            'MDVP:Fo(Hz)': np.mean(f0) if len(f0) > 0 else 0,
            'MDVP:Fhi(Hz)': np.max(f0) if len(f0) > 0 else 0,
            'MDVP:Flo(Hz)': np.min(f0) if len(f0) > 0 else 0,
            'MDVP:Jitter(%)': jitter_percent,
            'MDVP:Jitter(Abs)': jitter_abs,
            'rap' : np.mean(np.abs(np.diff(f0))),
            'ppq' : np.mean(np.abs(np.diff(f0, 2))),
            'ddp' : np.mean(np.abs(np.diff(f0))) * 3 ,
            'MDVP:Shimmer': shimmer,
            'MDVP:Shimmer(dB)': shimmer_db,  #negative value may indicate healthy voice
            'Shimmer:APQ3': apq3,
            'Shimmer:APQ5': apq5,
            'MDVP:APQ': mdvp_apq,
            'Shimmer:DDA': dda,
            'NHR' : np.mean(np.abs(percussive)) / np.mean(np.abs(harmonic)),
            'HNR' : np.mean(np.abs(harmonic)) / np.mean(np.abs(percussive)),
            # Add more features as needed
        }

        df = pd.DataFrame([features_dict])

        fig, ax = plt.subplots()

        for column in df.columns:
            ax.bar(column, df[column].values[0])


        ax.set_xlabel('Features')

        ax.set_ylabel('Value')

        ax.set_title('Values of Features')

        plt.xticks(rotation=90)

        plt.savefig('plot.png', bbox_inches='tight')
        # plt.show()

        image = Image.open('plot.png')

        st.write("Features Extracted Successfully!")

        if st.button("Get Output"):
            st.write("Wait a bit generating....")
            st.write("Extracted Features:")
            st.write(df)
            st.image(image, caption='Audio_features', use_column_width=True)

if __name__ == "__main__":
    main()