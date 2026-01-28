"""
Audio Signal Processing Web App
Streamlit-based UI for audio analysis and filtering
"""

import streamlit as st
import numpy as np
import soundfile as sf
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import os

import signal_processing as dsp


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Audio Signal Processor",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stat-box {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stat-value {
        font-size: 1.5em;
        font-weight: 600;
        color: #00d9ff;
    }
    .stat-label {
        font-size: 0.85em;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

st.title("Audio Signal Processor")


# =============================================================================
# Session State
# =============================================================================

if 'audio_left' not in st.session_state:
    st.session_state.audio_left = None
if 'audio_right' not in st.session_state:
    st.session_state.audio_right = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'original_left' not in st.session_state:
    st.session_state.original_left = None
if 'original_right' not in st.session_state:
    st.session_state.original_right = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'impedance_low' not in st.session_state:
    st.session_state.impedance_low = None
if 'impedance_mid' not in st.session_state:
    st.session_state.impedance_mid = None
if 'impedance_high' not in st.session_state:
    st.session_state.impedance_high = None


# =============================================================================
# File Upload
# =============================================================================

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'aif', 'aiff', 'flac', 'ogg', 'mp3'],
    help="Supported formats: WAV, AIF, AIFF, FLAC, OGG, MP3. Files longer than 60 seconds will be truncated."
)

# Maximum duration in seconds (to prevent memory issues on cloud)
MAX_DURATION_SECONDS = 240

if uploaded_file is not None:
    # Check file size first (warn if very large)
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size_mb > 50:
        st.warning(f"Large file detected ({file_size_mb:.1f} MB). Audio will be truncated to {MAX_DURATION_SECONDS} seconds to prevent memory issues.")
    
    # Load audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    try:
        audio_data, sample_rate = sf.read(tmp_path)
        
        # Handle mono vs stereo
        if len(audio_data.shape) == 1:
            # Mono: use same signal for left and right
            audio_left = audio_data
            audio_right = audio_data.copy()
        else:
            # Stereo: separate channels
            audio_left = audio_data[:, 0]
            audio_right = audio_data[:, 1]
        
        # Truncate to maximum duration to prevent memory issues
        max_samples = int(MAX_DURATION_SECONDS * sample_rate)
        original_duration = len(audio_left) / sample_rate
        if len(audio_left) > max_samples:
            audio_left = audio_left[:max_samples]
            audio_right = audio_right[:max_samples]
            st.info(f"Audio truncated from {original_duration:.1f}s to {MAX_DURATION_SECONDS}s for analysis.")
        
        # Truncate to start and end near zero crossings
        audio_left = dsp.truncate_to_zero_crossings(audio_left)
        audio_right = dsp.truncate_to_zero_crossings(audio_right)
        
        # Ensure same length (use shorter)
        min_len = min(len(audio_left), len(audio_right))
        audio_left = audio_left[:min_len]
        audio_right = audio_right[:min_len]
        
        st.session_state.audio_left = audio_left
        st.session_state.audio_right = audio_right
        st.session_state.original_left = audio_left.copy()
        st.session_state.original_right = audio_right.copy()
        st.session_state.sample_rate = sample_rate
        st.session_state.filename = uploaded_file.name
        
    finally:
        os.unlink(tmp_path)


# =============================================================================
# Main Content (only show if audio is loaded)
# =============================================================================

if st.session_state.audio_left is not None:
    audio_left = st.session_state.audio_left
    audio_right = st.session_state.audio_right
    sample_rate = st.session_state.sample_rate
    
    # -------------------------------------------------------------------------
    # Audio Playback
    # -------------------------------------------------------------------------
    st.subheader("Audio Playback")
    
    # Convert to stereo bytes for playback
    stereo_audio = np.column_stack((audio_left, audio_right))
    buffer = BytesIO()
    sf.write(buffer, stereo_audio, sample_rate, format='WAV')
    st.audio(buffer.getvalue(), format='audio/wav')
    
    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    st.subheader("Audio Statistics")
    
    stats_left = dsp.get_audio_stats(audio_left, sample_rate)
    stats_right = dsp.get_audio_stats(audio_right, sample_rate)
    
    st.markdown("**Left Channel**")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Duration", f"{stats_left['duration']:.2f} s")
    with col2:
        st.metric("Sample Rate", f"{stats_left['sample_rate']:,} Hz")
    with col3:
        st.metric("Samples", f"{stats_left['samples']:,}")
    with col4:
        st.metric("RMS Level", f"{stats_left['rms']:.4f}")
    with col5:
        st.metric("Peak Level", f"{stats_left['peak']:.4f}")
    with col6:
        st.metric("Crest Factor", f"{stats_left['crest_factor']:.2f}")
    
    st.markdown("**Right Channel**")
    col1r, col2r, col3r, col4r, col5r, col6r = st.columns(6)
    with col1r:
        st.metric("Duration", f"{stats_right['duration']:.2f} s")
    with col2r:
        st.metric("Sample Rate", f"{stats_right['sample_rate']:,} Hz")
    with col3r:
        st.metric("Samples", f"{stats_right['samples']:,}")
    with col4r:
        st.metric("RMS Level", f"{stats_right['rms']:.4f}")
    with col5r:
        st.metric("Peak Level", f"{stats_right['peak']:.4f}")
    with col6r:
        st.metric("Crest Factor", f"{stats_right['crest_factor']:.2f}")
    
    # -------------------------------------------------------------------------
    # Input Waveform
    # -------------------------------------------------------------------------
    st.subheader("Input Waveform")
    
    display_samples = 32000
    input_time_axis = np.linspace(0, len(audio_left) / sample_rate, display_samples)
    
    col_wave_l, col_wave_r = st.columns(2)
    
    with col_wave_l:
        st.markdown("**Left Channel**")
        input_display_left = dsp.downsample_for_display(audio_left, display_samples)
        fig_input_l = go.Figure()
        fig_input_l.add_trace(go.Scatter(
            x=input_time_axis,
            y=input_display_left,
            mode='lines',
            line=dict(color='#00d9ff', width=1),
            name='Left'
        ))
        fig_input_l.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=150,
            margin=dict(l=50, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_input_l, use_container_width=True, key="input_wave_l")
    
    with col_wave_r:
        st.markdown("**Right Channel**")
        input_display_right = dsp.downsample_for_display(audio_right, display_samples)
        fig_input_r = go.Figure()
        fig_input_r.add_trace(go.Scatter(
            x=input_time_axis,
            y=input_display_right,
            mode='lines',
            line=dict(color='#ff6b6b', width=1),
            name='Right'
        ))
        fig_input_r.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=150,
            margin=dict(l=50, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_input_r, use_container_width=True, key="input_wave_r")
    
    # -------------------------------------------------------------------------
    # Input Frequency Spectrum
    # -------------------------------------------------------------------------
    st.subheader("Input Frequency Spectrum")
    
    fft_size = min(len(audio_left), 16384)
    
    col_spec_l, col_spec_r = st.columns(2)
    
    with col_spec_l:
        st.markdown("**Left Channel**")
        fft_result_left = dsp.compute_fft(audio_left[:fft_size], sample_rate)
        power_db_left = dsp.compute_power_spectrum(fft_result_left['magnitudes'])
        freq_mask = (fft_result_left['frequencies'] >= 20) & (fft_result_left['frequencies'] <= sample_rate / 2)
        
        fig_spec_l = go.Figure()
        fig_spec_l.add_trace(go.Scatter(
            x=fft_result_left['frequencies'][freq_mask],
            y=power_db_left[freq_mask],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#00d9ff', width=1.5),
            fillcolor='rgba(0, 217, 255, 0.3)'
        ))
        fig_spec_l.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            xaxis_type="log",
            template="plotly_dark",
            height=150,
            margin=dict(l=50, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_l, use_container_width=True, key="input_spec_l")
    
    with col_spec_r:
        st.markdown("**Right Channel**")
        fft_result_right = dsp.compute_fft(audio_right[:fft_size], sample_rate)
        power_db_right = dsp.compute_power_spectrum(fft_result_right['magnitudes'])
        
        fig_spec_r = go.Figure()
        fig_spec_r.add_trace(go.Scatter(
            x=fft_result_right['frequencies'][freq_mask],
            y=power_db_right[freq_mask],
            mode='lines',
            fill='tozeroy',
            line=dict(color='#ff6b6b', width=1.5),
            fillcolor='rgba(255, 107, 107, 0.3)'
        ))
        fig_spec_r.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="Power (dB)",
            xaxis_type="log",
            template="plotly_dark",
            height=150,
            margin=dict(l=50, r=20, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_r, use_container_width=True, key="input_spec_r")
    
    # -------------------------------------------------------------------------
    # Filter Controls
    # -------------------------------------------------------------------------
    st.subheader("Signal Processing Controls")
    
    col_filter, col_params, col_actions = st.columns([1, 2, 1])
    
    with col_filter:
        filter_type = st.selectbox(
            "Filter Type",
            ["None", "Low-Pass", "High-Pass", "Band-Pass"]
        )
    
    with col_params:
        if filter_type == "Band-Pass":
            param_col1, param_col2 = st.columns(2)
            with param_col1:
                low_freq = st.number_input("Low Cutoff (Hz)", min_value=20, max_value=20000, value=200)
            with param_col2:
                high_freq = st.number_input("High Cutoff (Hz)", min_value=20, max_value=20000, value=5000)
        elif filter_type != "None":
            cutoff_freq = st.number_input("Cutoff Frequency (Hz)", min_value=20, max_value=20000, value=1000)
    
    with col_actions:
        st.write("")  # Spacing
        st.write("")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            apply_clicked = st.button("Apply Filter", type="primary")
        with col_btn2:
            reset_clicked = st.button("Reset")
    
    # Apply filter
    if apply_clicked and filter_type != "None":
        if filter_type == "Low-Pass":
            st.session_state.audio_left = dsp.low_pass_filter(
                st.session_state.original_left, cutoff_freq, sample_rate
            )
            st.session_state.audio_right = dsp.low_pass_filter(
                st.session_state.original_right, cutoff_freq, sample_rate
            )
        elif filter_type == "High-Pass":
            st.session_state.audio_left = dsp.high_pass_filter(
                st.session_state.original_left, cutoff_freq, sample_rate
            )
            st.session_state.audio_right = dsp.high_pass_filter(
                st.session_state.original_right, cutoff_freq, sample_rate
            )
        elif filter_type == "Band-Pass":
            st.session_state.audio_left = dsp.band_pass_filter(
                st.session_state.original_left, low_freq, high_freq, sample_rate
            )
            st.session_state.audio_right = dsp.band_pass_filter(
                st.session_state.original_right, low_freq, high_freq, sample_rate
            )
        st.rerun()
    
    # Reset
    if reset_clicked:
        st.session_state.audio_left = st.session_state.original_left.copy()
        st.session_state.audio_right = st.session_state.original_right.copy()
        st.rerun()
    
    # -------------------------------------------------------------------------
    # 3-Way Crossover Configuration
    # -------------------------------------------------------------------------
    st.subheader("3-Way Crossover")
    
    col_xover1, col_xover2, col_xover3, col_xover4 = st.columns(4)
    
    with col_xover1:
        low_band_hpf = st.slider(
            "Low Band HPF (Hz)",
            min_value=20,
            max_value=100,
            value=60,
            step=5,
            help="Additional 24dB/oct highpass filter on low band output"
        )
    
    with col_xover2:
        low_mid_freq = st.slider(
            "Low/Mid Crossover (Hz)",
            min_value=20,
            max_value=2000,
            value=200,
            step=10,
            help="Crossover frequency between low and mid bands"
        )
    
    with col_xover3:
        mid_high_freq = st.slider(
            "Mid/High Crossover (Hz)",
            min_value=500,
            max_value=20000,
            value=2000,
            step=100,
            help="Crossover frequency between mid and high bands"
        )
    
    with col_xover4:
        slope_options = {
            "12 dB/oct (2nd order)": 2,
            "24 dB/oct (4th order)": 4,
            "36 dB/oct (6th order)": 6,
            "48 dB/oct (8th order)": 8
        }
        slope_selection = st.selectbox(
            "Filter Slope",
            options=list(slope_options.keys()),
            index=1,
            help="Steepness of the crossover filters"
        )
        filter_order = slope_options[slope_selection]
    
    # Get crossover frequency responses
    xover_responses = dsp.get_crossover_responses(
        low_mid_freq, mid_high_freq, sample_rate, filter_order, low_band_hpf
    )
    
    # Plot crossover response
    fig_xover = go.Figure()
    
    fig_xover.add_trace(go.Scatter(
        x=xover_responses['frequencies'],
        y=xover_responses['low_db'],
        mode='lines',
        line=dict(color='#e74c3c', width=2),
        name='Low'
    ))
    
    fig_xover.add_trace(go.Scatter(
        x=xover_responses['frequencies'],
        y=xover_responses['mid_db'],
        mode='lines',
        line=dict(color='#2ecc71', width=2),
        name='Mid'
    ))
    
    fig_xover.add_trace(go.Scatter(
        x=xover_responses['frequencies'],
        y=xover_responses['high_db'],
        mode='lines',
        line=dict(color='#3498db', width=2),
        name='High'
    ))
    
    # Add crossover frequency markers
    fig_xover.add_vline(x=low_mid_freq, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_xover.add_vline(x=mid_high_freq, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    # Add low band HPF marker
    fig_xover.add_vline(x=low_band_hpf, line_dash="dot", line_color="rgba(231,76,60,0.5)",
                        annotation_text="HPF", annotation_position="top left")
    
    fig_xover.update_layout(
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (dB)",
        xaxis_type="log",
        template="plotly_dark",
        height=250,
        margin=dict(l=50, r=20, t=20, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        xaxis=dict(range=[np.log10(20), np.log10(20000)]),
        yaxis=dict(range=[-60, 5]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_xover, use_container_width=True, key="xover_response")
    
    # -------------------------------------------------------------------------
    # Crossover Band Waveforms
    # -------------------------------------------------------------------------
    st.subheader("Crossover Band Waveforms")
    
    # Compute crossover bands for both channels
    bands_left = dsp.three_way_crossover(
        st.session_state.original_left,
        low_mid_freq,
        mid_high_freq,
        sample_rate,
        filter_order,
        low_band_hpf
    )
    bands_right = dsp.three_way_crossover(
        st.session_state.original_right,
        low_mid_freq,
        mid_high_freq,
        sample_rate,
        filter_order,
        low_band_hpf
    )
    
    # Downsample for display
    display_samples = 32000
    time_axis = np.linspace(0, len(audio_left) / sample_rate, display_samples)
    
    # Left channel display
    st.markdown("**Left Channel**")
    col_low_l, col_mid_l, col_high_l = st.columns(3)
    
    with col_low_l:
        st.markdown("*Low Band*")
        low_display_l = dsp.downsample_for_display(bands_left['low'], display_samples)
        fig_low_l = go.Figure()
        fig_low_l.add_trace(go.Scatter(
            x=time_axis, y=low_display_l,
            mode='lines', line=dict(color='#e74c3c', width=1)
        ))
        fig_low_l.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_low_l, use_container_width=True, key="low_wave_l")
        st.text(f"RMS: {dsp.calculate_rms(bands_left['low']):.4f}  |  Peak: {dsp.calculate_peak(bands_left['low']):.4f}")
        
        # Spectrum
        fft_low_l = dsp.compute_fft(bands_left['low'][:fft_size], sample_rate)
        power_low_l = dsp.compute_power_spectrum(fft_low_l['magnitudes'])
        fig_spec_low_l = go.Figure()
        fig_spec_low_l.add_trace(go.Scatter(
            x=fft_low_l['frequencies'][freq_mask], y=power_low_l[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#e74c3c', width=1), fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        fig_spec_low_l.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_low_l, use_container_width=True, key="low_spec_l")
    
    with col_mid_l:
        st.markdown("*Mid Band*")
        mid_display_l = dsp.downsample_for_display(bands_left['mid'], display_samples)
        fig_mid_l = go.Figure()
        fig_mid_l.add_trace(go.Scatter(
            x=time_axis, y=mid_display_l,
            mode='lines', line=dict(color='#2ecc71', width=1)
        ))
        fig_mid_l.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_mid_l, use_container_width=True, key="mid_wave_l")
        st.text(f"RMS: {dsp.calculate_rms(bands_left['mid']):.4f}  |  Peak: {dsp.calculate_peak(bands_left['mid']):.4f}")
        
        # Spectrum
        fft_mid_l = dsp.compute_fft(bands_left['mid'][:fft_size], sample_rate)
        power_mid_l = dsp.compute_power_spectrum(fft_mid_l['magnitudes'])
        fig_spec_mid_l = go.Figure()
        fig_spec_mid_l.add_trace(go.Scatter(
            x=fft_mid_l['frequencies'][freq_mask], y=power_mid_l[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#2ecc71', width=1), fillcolor='rgba(46, 204, 113, 0.3)'
        ))
        fig_spec_mid_l.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_mid_l, use_container_width=True, key="mid_spec_l")
    
    with col_high_l:
        st.markdown("*High Band*")
        high_display_l = dsp.downsample_for_display(bands_left['high'], display_samples)
        fig_high_l = go.Figure()
        fig_high_l.add_trace(go.Scatter(
            x=time_axis, y=high_display_l,
            mode='lines', line=dict(color='#3498db', width=1)
        ))
        fig_high_l.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_high_l, use_container_width=True, key="high_wave_l")
        st.text(f"RMS: {dsp.calculate_rms(bands_left['high']):.4f}  |  Peak: {dsp.calculate_peak(bands_left['high']):.4f}")
        
        # Spectrum
        fft_high_l = dsp.compute_fft(bands_left['high'][:fft_size], sample_rate)
        power_high_l = dsp.compute_power_spectrum(fft_high_l['magnitudes'])
        fig_spec_high_l = go.Figure()
        fig_spec_high_l.add_trace(go.Scatter(
            x=fft_high_l['frequencies'][freq_mask], y=power_high_l[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#3498db', width=1), fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        fig_spec_high_l.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_high_l, use_container_width=True, key="high_spec_l")
    
    # Right channel display
    st.markdown("**Right Channel**")
    col_low_r, col_mid_r, col_high_r = st.columns(3)
    
    with col_low_r:
        st.markdown("*Low Band*")
        low_display_r = dsp.downsample_for_display(bands_right['low'], display_samples)
        fig_low_r = go.Figure()
        fig_low_r.add_trace(go.Scatter(
            x=time_axis, y=low_display_r,
            mode='lines', line=dict(color='#e74c3c', width=1)
        ))
        fig_low_r.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_low_r, use_container_width=True, key="low_wave_r")
        st.text(f"RMS: {dsp.calculate_rms(bands_right['low']):.4f}  |  Peak: {dsp.calculate_peak(bands_right['low']):.4f}")
        
        # Spectrum
        fft_low_r = dsp.compute_fft(bands_right['low'][:fft_size], sample_rate)
        power_low_r = dsp.compute_power_spectrum(fft_low_r['magnitudes'])
        fig_spec_low_r = go.Figure()
        fig_spec_low_r.add_trace(go.Scatter(
            x=fft_low_r['frequencies'][freq_mask], y=power_low_r[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#e74c3c', width=1), fillcolor='rgba(231, 76, 60, 0.3)'
        ))
        fig_spec_low_r.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_low_r, use_container_width=True, key="low_spec_r")
    
    with col_mid_r:
        st.markdown("*Mid Band*")
        mid_display_r = dsp.downsample_for_display(bands_right['mid'], display_samples)
        fig_mid_r = go.Figure()
        fig_mid_r.add_trace(go.Scatter(
            x=time_axis, y=mid_display_r,
            mode='lines', line=dict(color='#2ecc71', width=1)
        ))
        fig_mid_r.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_mid_r, use_container_width=True, key="mid_wave_r")
        st.text(f"RMS: {dsp.calculate_rms(bands_right['mid']):.4f}  |  Peak: {dsp.calculate_peak(bands_right['mid']):.4f}")
        
        # Spectrum
        fft_mid_r = dsp.compute_fft(bands_right['mid'][:fft_size], sample_rate)
        power_mid_r = dsp.compute_power_spectrum(fft_mid_r['magnitudes'])
        fig_spec_mid_r = go.Figure()
        fig_spec_mid_r.add_trace(go.Scatter(
            x=fft_mid_r['frequencies'][freq_mask], y=power_mid_r[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#2ecc71', width=1), fillcolor='rgba(46, 204, 113, 0.3)'
        ))
        fig_spec_mid_r.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_mid_r, use_container_width=True, key="mid_spec_r")
    
    with col_high_r:
        st.markdown("*High Band*")
        high_display_r = dsp.downsample_for_display(bands_right['high'], display_samples)
        fig_high_r = go.Figure()
        fig_high_r.add_trace(go.Scatter(
            x=time_axis, y=high_display_r,
            mode='lines', line=dict(color='#3498db', width=1)
        ))
        fig_high_r.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_high_r, use_container_width=True, key="high_wave_r")
        st.text(f"RMS: {dsp.calculate_rms(bands_right['high']):.4f}  |  Peak: {dsp.calculate_peak(bands_right['high']):.4f}")
        
        # Spectrum
        fft_high_r = dsp.compute_fft(bands_right['high'][:fft_size], sample_rate)
        power_high_r = dsp.compute_power_spectrum(fft_high_r['magnitudes'])
        fig_spec_high_r = go.Figure()
        fig_spec_high_r.add_trace(go.Scatter(
            x=fft_high_r['frequencies'][freq_mask], y=power_high_r[freq_mask],
            mode='lines', fill='tozeroy',
            line=dict(color='#3498db', width=1), fillcolor='rgba(52, 152, 219, 0.3)'
        ))
        fig_spec_high_r.update_layout(
            xaxis_title="Frequency (Hz)", yaxis_title="Power (dB)", xaxis_type="log",
            template="plotly_dark", height=120,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-80, 0])
        )
        st.plotly_chart(fig_spec_high_r, use_container_width=True, key="high_spec_r")
    
    # -------------------------------------------------------------------------
    # Crossover Check (sum of band outputs)
    # -------------------------------------------------------------------------
    st.subheader("Crossover Check")
    st.caption("Sum of low + mid + high bands - should match original input")
    
    # Sum the bands
    summed_left = bands_left['low'] + bands_left['mid'] + bands_left['high']
    summed_right = bands_right['low'] + bands_right['mid'] + bands_right['high']
    
    col_check_l, col_check_r = st.columns(2)
    
    with col_check_l:
        st.markdown("**Left Channel**")
        summed_display_l = dsp.downsample_for_display(summed_left, display_samples)
        fig_sum_l = go.Figure()
        fig_sum_l.add_trace(go.Scatter(
            x=time_axis, y=summed_display_l,
            mode='lines', line=dict(color='#9b59b6', width=1),
            name='Summed'
        ))
        fig_sum_l.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_sum_l, use_container_width=True, key="summed_wave_l")
        st.text(f"RMS: {dsp.calculate_rms(summed_left):.4f}  |  Peak: {dsp.calculate_peak(summed_left):.4f}")
    
    with col_check_r:
        st.markdown("**Right Channel**")
        summed_display_r = dsp.downsample_for_display(summed_right, display_samples)
        fig_sum_r = go.Figure()
        fig_sum_r.add_trace(go.Scatter(
            x=time_axis, y=summed_display_r,
            mode='lines', line=dict(color='#e67e22', width=1),
            name='Summed'
        ))
        fig_sum_r.update_layout(
            xaxis_title="Time (s)", yaxis_title="Amplitude",
            template="plotly_dark", height=150,
            margin=dict(l=50, r=10, t=10, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
            yaxis=dict(range=[-1, 1], autorange=False)
        )
        st.plotly_chart(fig_sum_r, use_container_width=True, key="summed_wave_r")
        st.text(f"RMS: {dsp.calculate_rms(summed_right):.4f}  |  Peak: {dsp.calculate_peak(summed_right):.4f}")
    
    # -------------------------------------------------------------------------
    # Speaker Amplifier (Full-Bridge / H-Bridge)
    # -------------------------------------------------------------------------
    st.subheader("Speaker Amplifier")
    st.caption("Full-bridge (H-bridge) configuration: output swings from +Vbus to -Vbus")
    
    col_amp1, col_amp2 = st.columns(2)
    
    with col_amp1:
        bus_voltage = st.number_input(
            "Bus Voltage (V)",
            min_value=0.1,
            max_value=100.0,
            value=35.0,
            step=0.1,
            help="Full-scale range voltage for amplifier"
        )
    
    with col_amp2:
        volume_percent = st.slider(
            "Volume (%)",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            help="Volume level applied to crossover outputs"
        )
    
    volume_scale = volume_percent / 100.0
    
    # -------------------------------------------------------------------------
    # Transducer
    # -------------------------------------------------------------------------
    st.subheader("Transducer")
    st.caption("Upload impedance CSV files (frequency, impedance, phase) or use simple resistance values")
    
    col_trans1, col_trans2, col_trans3 = st.columns(3)
    
    with col_trans1:
        st.markdown("*Low Band*")
        impedance_file_low = st.file_uploader(
            "Impedance CSV",
            type=['csv'],
            key="imp_low",
            help="CSV with columns: frequency, impedance, phase"
        )
        if impedance_file_low is not None:
            try:
                st.session_state.impedance_low = dsp.load_impedance_csv(impedance_file_low.getvalue())
                st.success(f"Loaded: {len(st.session_state.impedance_low['frequencies'])} points")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.session_state.impedance_low = None
        else:
            st.session_state.impedance_low = None
        
        use_impedance_low = st.session_state.impedance_low is not None
        resistance_low = st.number_input(
            "Fallback Resistance (Ohms)",
            min_value=0.1,
            max_value=100.0,
            value=4.0,
            step=0.1,
            key="res_low",
            help="Used if no impedance CSV loaded",
            disabled=use_impedance_low
        )
        if use_impedance_low:
            imp_data = st.session_state.impedance_low
            fig_imp_low = go.Figure()
            fig_imp_low.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['impedance'],
                mode='lines+markers', line=dict(color='#e74c3c', width=1.5),
                marker=dict(size=3), name='|Z|'
            ))
            fig_imp_low.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['phase'],
                mode='lines+markers', line=dict(color='#e74c3c', width=1.5, dash='dot'),
                marker=dict(size=3), name='Phase', yaxis='y2'
            ))
            fig_imp_low.update_layout(
                xaxis_title="Freq (Hz)", yaxis_title="|Z| (Ohms)",
                yaxis2=dict(title="Phase (deg)", overlaying='y', side='right'),
                xaxis_type="log", template="plotly_dark", height=150,
                margin=dict(l=50, r=50, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True
            )
            st.plotly_chart(fig_imp_low, use_container_width=True, key="impedance_low")
    
    with col_trans2:
        st.markdown("*Mid Band*")
        impedance_file_mid = st.file_uploader(
            "Impedance CSV",
            type=['csv'],
            key="imp_mid",
            help="CSV with columns: frequency, impedance, phase"
        )
        if impedance_file_mid is not None:
            try:
                st.session_state.impedance_mid = dsp.load_impedance_csv(impedance_file_mid.getvalue())
                st.success(f"Loaded: {len(st.session_state.impedance_mid['frequencies'])} points")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.session_state.impedance_mid = None
        else:
            st.session_state.impedance_mid = None
        
        use_impedance_mid = st.session_state.impedance_mid is not None
        resistance_mid = st.number_input(
            "Fallback Resistance (Ohms)",
            min_value=0.1,
            max_value=100.0,
            value=4.0,
            step=0.1,
            key="res_mid",
            help="Used if no impedance CSV loaded",
            disabled=use_impedance_mid
        )
        if use_impedance_mid:
            imp_data = st.session_state.impedance_mid
            fig_imp_mid = go.Figure()
            fig_imp_mid.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['impedance'],
                mode='lines+markers', line=dict(color='#2ecc71', width=1.5),
                marker=dict(size=3), name='|Z|'
            ))
            fig_imp_mid.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['phase'],
                mode='lines+markers', line=dict(color='#2ecc71', width=1.5, dash='dot'),
                marker=dict(size=3), name='Phase', yaxis='y2'
            ))
            fig_imp_mid.update_layout(
                xaxis_title="Freq (Hz)", yaxis_title="|Z| (Ohms)",
                yaxis2=dict(title="Phase (deg)", overlaying='y', side='right'),
                xaxis_type="log", template="plotly_dark", height=150,
                margin=dict(l=50, r=50, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True
            )
            st.plotly_chart(fig_imp_mid, use_container_width=True, key="impedance_mid")
    
    with col_trans3:
        st.markdown("*High Band*")
        impedance_file_high = st.file_uploader(
            "Impedance CSV",
            type=['csv'],
            key="imp_high",
            help="CSV with columns: frequency, impedance, phase"
        )
        if impedance_file_high is not None:
            try:
                st.session_state.impedance_high = dsp.load_impedance_csv(impedance_file_high.getvalue())
                st.success(f"Loaded: {len(st.session_state.impedance_high['frequencies'])} points")
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
                st.session_state.impedance_high = None
        else:
            st.session_state.impedance_high = None
        
        use_impedance_high = st.session_state.impedance_high is not None
        resistance_high = st.number_input(
            "Fallback Resistance (Ohms)",
            min_value=0.1,
            max_value=100.0,
            value=4.0,
            step=0.1,
            key="res_high",
            help="Used if no impedance CSV loaded",
            disabled=use_impedance_high
        )
        if use_impedance_high:
            imp_data = st.session_state.impedance_high
            fig_imp_high = go.Figure()
            fig_imp_high.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['impedance'],
                mode='lines+markers', line=dict(color='#3498db', width=1.5),
                marker=dict(size=3), name='|Z|'
            ))
            fig_imp_high.add_trace(go.Scatter(
                x=imp_data['frequencies'], y=imp_data['phase'],
                mode='lines+markers', line=dict(color='#3498db', width=1.5, dash='dot'),
                marker=dict(size=3), name='Phase', yaxis='y2'
            ))
            fig_imp_high.update_layout(
                xaxis_title="Freq (Hz)", yaxis_title="|Z| (Ohms)",
                yaxis2=dict(title="Phase (deg)", overlaying='y', side='right'),
                xaxis_type="log", template="plotly_dark", height=150,
                margin=dict(l=50, r=50, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.3)',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                showlegend=True
            )
            st.plotly_chart(fig_imp_high, use_container_width=True, key="impedance_high")
    
    # -------------------------------------------------------------------------
    # Acoustics - Thiele/Small Parameters
    # -------------------------------------------------------------------------
    st.subheader("Acoustics - Thiele/Small Parameters")
    st.caption("Calculate T/S parameters from impedance curves")
    
    # Check which bands have impedance data
    has_imp_low = st.session_state.impedance_low is not None
    has_imp_mid = st.session_state.impedance_mid is not None
    has_imp_high = st.session_state.impedance_high is not None
    
    # Initialize eq_settings with defaults (will be overwritten if impedance data exists)
    eq_settings = {
        'Low': [{'freq': 100, 'q': 1.0, 'gain': 0}, {'freq': 200, 'q': 1.0, 'gain': 0}, {'freq': 300, 'q': 1.0, 'gain': 0}],
        'Mid': [{'freq': 500, 'q': 1.0, 'gain': 0}, {'freq': 1000, 'q': 1.0, 'gain': 0}, {'freq': 2000, 'q': 1.0, 'gain': 0}],
        'High': [{'freq': 5000, 'q': 1.0, 'gain': 0}, {'freq': 10000, 'q': 1.0, 'gain': 0}, {'freq': 15000, 'q': 1.0, 'gain': 0}]
    }
    
    if not (has_imp_low or has_imp_mid or has_imp_high):
        st.info("Load impedance data for at least one band to calculate Thiele/Small parameters.")
    else:
        # User inputs for each band with impedance data
        ts_cols = st.columns(3)
        
        ts_params_low = None
        ts_params_mid = None
        ts_params_high = None
        
        with ts_cols[0]:
            st.markdown("**Low Band**")
            if has_imp_low:
                sd_low = st.number_input(
                    "Sd (cm²)", 
                    min_value=0.1, max_value=1000.0, value=543.0, step=1.0,
                    key="sd_low",
                    help="Effective diaphragm area"
                )
                mmd_low = st.number_input(
                    "Mmd (grams)", 
                    min_value=0.1, max_value=500.0, value=200.0, step=0.5,
                    key="mmd_low",
                    help="Moving mass (diaphragm + voice coil)"
                )
                # Convert units: cm² to m², grams to kg
                ts_params_low = dsp.calculate_thiele_small_parameters(
                    st.session_state.impedance_low,
                    sd_low * 1e-4,  # cm² to m²
                    mmd_low * 1e-3  # grams to kg
                )
            else:
                st.caption("No impedance data")
        
        with ts_cols[1]:
            st.markdown("**Mid Band**")
            if has_imp_mid:
                sd_mid = st.number_input(
                    "Sd (cm²)", 
                    min_value=0.1, max_value=1000.0, value=62.0, step=1.0,
                    key="sd_mid",
                    help="Effective diaphragm area"
                )
                mmd_mid = st.number_input(
                    "Mmd (grams)", 
                    min_value=0.01, max_value=100.0, value=4.5, step=0.1,
                    key="mmd_mid",
                    help="Moving mass (diaphragm + voice coil)"
                )
                ts_params_mid = dsp.calculate_thiele_small_parameters(
                    st.session_state.impedance_mid,
                    sd_mid * 1e-4,
                    mmd_mid * 1e-3
                )
            else:
                st.caption("No impedance data")
        
        with ts_cols[2]:
            st.markdown("**High Band**")
            if has_imp_high:
                sd_high = st.number_input(
                    "Sd (cm²)", 
                    min_value=0.01, max_value=100.0, value=5.0, step=0.5,
                    key="sd_high",
                    help="Effective diaphragm area"
                )
                mmd_high = st.number_input(
                    "Mmd (grams)", 
                    min_value=0.001, max_value=10.0, value=0.5, step=0.05,
                    key="mmd_high",
                    help="Moving mass (diaphragm + voice coil)"
                )
                ts_params_high = dsp.calculate_thiele_small_parameters(
                    st.session_state.impedance_high,
                    sd_high * 1e-4,
                    mmd_high * 1e-3
                )
            else:
                st.caption("No impedance data")
        
        # Display calculated parameters
        st.markdown("---")
        st.markdown("**Calculated Parameters**")
        
        # Create a table of all parameters
        param_labels = {
            'fs': ('Fs', 'Hz', 'Resonant frequency'),
            're': ('Re', 'Ω', 'DC resistance'),
            'qts': ('Qts', '', 'Total Q factor'),
            'qes': ('Qes', '', 'Electrical Q'),
            'qms': ('Qms', '', 'Mechanical Q'),
            'vas': ('Vas', 'L', 'Equivalent volume'),
            'mms': ('Mms', 'g', 'Total moving mass'),
            'cms': ('Cms', 'mm/N', 'Compliance'),
            'bl': ('BL', 'Tm', 'Force factor'),
            'spl_1w1m': ('SPL', 'dB', 'Sensitivity (1W/1m)'),
            'eta0': ('η₀', '%', 'Efficiency'),
            'sd': ('Sd', 'cm²', 'Diaphragm area'),
        }
        
        result_cols = st.columns(3)
        
        for idx, (ts_params, band_name) in enumerate([
            (ts_params_low, "Low"),
            (ts_params_mid, "Mid"),
            (ts_params_high, "High")
        ]):
            with result_cols[idx]:
                if ts_params is not None:
                    st.markdown(f"**{band_name} Band Results**")
                    
                    # Display key parameters in a compact format
                    for key, (label, unit, desc) in param_labels.items():
                        if key in ts_params:
                            value = ts_params[key]
                            if isinstance(value, float):
                                if abs(value) >= 100:
                                    formatted = f"{value:.1f}"
                                elif abs(value) >= 10:
                                    formatted = f"{value:.2f}"
                                elif abs(value) >= 1:
                                    formatted = f"{value:.3f}"
                                else:
                                    formatted = f"{value:.4f}"
                            else:
                                formatted = str(value)
                            st.markdown(f"<span style='color:#888'>{label}:</span> **{formatted}** {unit}", 
                                       unsafe_allow_html=True)
        
        # Parametric EQ Section - 3-band EQ per driver
        st.markdown("---")
        st.markdown("**Parametric EQ (3-band per driver)**")
        st.caption("Applied to SPL response and power analysis - use to flatten individual driver responses")
        
        # Helper function to create 3-band EQ controls for a driver
        def create_3band_eq_controls(band_name, color, freq_range, key_prefix):
            """Create 3 parametric EQ bands for a driver"""
            st.markdown(f"<span style='color:{color}'>**{band_name} Driver EQ**</span>", unsafe_allow_html=True)
            
            eq_bands = []
            for i in range(3):
                with st.expander(f"Band {i+1}", expanded=(i==0)):
                    cols = st.columns(3)
                    with cols[0]:
                        freq = st.slider(
                            "Freq (Hz)",
                            min_value=freq_range[0],
                            max_value=freq_range[1],
                            value=freq_range[0] + (freq_range[1] - freq_range[0]) * (i + 1) // 4,
                            step=max(1, (freq_range[1] - freq_range[0]) // 100),
                            key=f"{key_prefix}_freq_{i}",
                        )
                    with cols[1]:
                        q = st.slider(
                            "Q",
                            min_value=0.1,
                            max_value=10.0,
                            value=1.0,
                            step=0.1,
                            key=f"{key_prefix}_q_{i}",
                        )
                    with cols[2]:
                        gain = st.slider(
                            "Gain (dB)",
                            min_value=-20.0,
                            max_value=20.0,
                            value=0.0,
                            step=0.5,
                            key=f"{key_prefix}_gain_{i}",
                        )
                    eq_bands.append({'freq': freq, 'q': q, 'gain': gain})
            return eq_bands
        
        eq_driver_cols = st.columns(3)
        
        with eq_driver_cols[0]:
            eq_low_bands = create_3band_eq_controls("Low", "#e74c3c", (20, 500), "eq_low")
        
        with eq_driver_cols[1]:
            eq_mid_bands = create_3band_eq_controls("Mid", "#2ecc71", (100, 5000), "eq_mid")
        
        with eq_driver_cols[2]:
            eq_high_bands = create_3band_eq_controls("High", "#3498db", (1000, 20000), "eq_high")
        
        # Store all EQ settings
        eq_settings = {
            'Low': eq_low_bands,
            'Mid': eq_mid_bands,
            'High': eq_high_bands
        }
        
        # Show combined EQ frequency response per driver
        fig_eq = go.Figure()
        colors_eq = {'Low': '#e74c3c', 'Mid': '#2ecc71', 'High': '#3498db'}
        
        any_eq_active = False
        for band_name, bands in eq_settings.items():
            # Calculate combined response for all 3 EQ bands
            combined_db = None
            for eq_band in bands:
                if eq_band['gain'] != 0:
                    any_eq_active = True
                    eq_response = dsp.get_parametric_eq_response(
                        eq_band['freq'], eq_band['q'], eq_band['gain'], sample_rate
                    )
                    if combined_db is None:
                        combined_db = np.array(eq_response['magnitude_db'])
                        frequencies = eq_response['frequencies']
                    else:
                        combined_db += np.array(eq_response['magnitude_db'])
            
            if combined_db is not None:
                fig_eq.add_trace(go.Scatter(
                    x=frequencies,
                    y=combined_db,
                    mode='lines',
                    line=dict(color=colors_eq[band_name], width=2),
                    name=f'{band_name} EQ'
                ))
        
        if any_eq_active:
            fig_eq.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_eq.update_layout(
                xaxis_title="Frequency (Hz)",
                yaxis_title="Gain (dB)",
                xaxis_type="log",
                template="plotly_dark",
                height=150,
                margin=dict(l=50, r=20, t=10, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.3)',
                xaxis=dict(range=[np.log10(20), np.log10(20000)]),
                yaxis=dict(range=[-25, 25]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_eq, use_container_width=True, key="eq_response")
    
    # -------------------------------------------------------------------------
    # Audio Power Analysis
    # -------------------------------------------------------------------------
    st.subheader("Audio Power Analysis")
    
    # Compute crossover bands first (without EQ)
    bands_left_raw = dsp.three_way_crossover(
        st.session_state.original_left,
        low_mid_freq,
        mid_high_freq,
        sample_rate,
        filter_order,
        low_band_hpf
    )
    bands_right_raw = dsp.three_way_crossover(
        st.session_state.original_right,
        low_mid_freq,
        mid_high_freq,
        sample_rate,
        filter_order,
        low_band_hpf
    )
    
    # Apply per-band EQ (3 bands each) to crossover outputs
    try:
        eq_active_count = {'Low': 0, 'Mid': 0, 'High': 0}
        
        # Helper to apply multiple EQ bands to a signal
        def apply_multi_band_eq(signal, eq_bands, sr):
            result = signal.copy()
            for eq_band in eq_bands:
                if eq_band['gain'] != 0:
                    result = dsp.parametric_eq(result, sr, eq_band['freq'], eq_band['q'], eq_band['gain'])
            return result
        
        # Apply all EQ bands for Low driver
        bands_left_low_eq = apply_multi_band_eq(bands_left_raw['low'], eq_settings['Low'], sample_rate)
        bands_right_low_eq = apply_multi_band_eq(bands_right_raw['low'], eq_settings['Low'], sample_rate)
        eq_active_count['Low'] = sum(1 for eq in eq_settings['Low'] if eq['gain'] != 0)
        
        # Apply all EQ bands for Mid driver
        bands_left_mid_eq = apply_multi_band_eq(bands_left_raw['mid'], eq_settings['Mid'], sample_rate)
        bands_right_mid_eq = apply_multi_band_eq(bands_right_raw['mid'], eq_settings['Mid'], sample_rate)
        eq_active_count['Mid'] = sum(1 for eq in eq_settings['Mid'] if eq['gain'] != 0)
        
        # Apply all EQ bands for High driver
        bands_left_high_eq = apply_multi_band_eq(bands_left_raw['high'], eq_settings['High'], sample_rate)
        bands_right_high_eq = apply_multi_band_eq(bands_right_raw['high'], eq_settings['High'], sample_rate)
        eq_active_count['High'] = sum(1 for eq in eq_settings['High'] if eq['gain'] != 0)
        
        # Reconstruct bands dict with EQ applied
        bands_left_eq = {'low': bands_left_low_eq, 'mid': bands_left_mid_eq, 'high': bands_left_high_eq}
        bands_right_eq = {'low': bands_right_low_eq, 'mid': bands_right_mid_eq, 'high': bands_right_high_eq}
        
        active_summary = [f"{k}: {v} bands" for k, v in eq_active_count.items() if v > 0]
        if active_summary:
            st.caption(f"EQ applied: {' | '.join(active_summary)}")
    
    except NameError:
        # EQ controls not yet rendered (no impedance data)
        bands_left_eq = bands_left_raw
        bands_right_eq = bands_right_raw
    
    # Helper function for power calculations with impedance support
    def calculate_power_metrics(bands, bus_voltage, volume,
                                 res_low, res_mid, res_high,
                                 imp_low, imp_mid, imp_high, sr):
        # Voltage signals (apply volume scaling)
        v_low = bands['low'] * volume * bus_voltage
        v_mid = bands['mid'] * volume * bus_voltage
        v_high = bands['high'] * volume * bus_voltage
        
        # Calculate power for each band using impedance if available
        if imp_low is not None:
            metrics_low = dsp.compute_power_with_impedance(v_low, sr, imp_low)
        else:
            metrics_low = dsp.compute_power_with_resistance(v_low, res_low, sr)
        
        if imp_mid is not None:
            metrics_mid = dsp.compute_power_with_impedance(v_mid, sr, imp_mid)
        else:
            metrics_mid = dsp.compute_power_with_resistance(v_mid, res_mid, sr)
        
        if imp_high is not None:
            metrics_high = dsp.compute_power_with_impedance(v_high, sr, imp_high)
        else:
            metrics_high = dsp.compute_power_with_resistance(v_high, res_high, sr)
        
        return {
            'low': metrics_low,
            'mid': metrics_mid,
            'high': metrics_high
        }
    
    # Calculate for both channels using EQ'd bands
    metrics_left = calculate_power_metrics(
        bands_left_eq, bus_voltage, volume_scale,
        resistance_low, resistance_mid, resistance_high,
        st.session_state.impedance_low, st.session_state.impedance_mid, 
        st.session_state.impedance_high, sample_rate
    )
    metrics_right = calculate_power_metrics(
        bands_right_eq, bus_voltage, volume_scale,
        resistance_low, resistance_mid, resistance_high,
        st.session_state.impedance_low, st.session_state.impedance_mid,
        st.session_state.impedance_high, sample_rate
    )
    
    # Display Left Channel metrics
    st.markdown("**Left Channel**")
    col_pwr1_l, col_pwr2_l, col_pwr3_l = st.columns(3)
    
    with col_pwr1_l:
        st.markdown("*Low Band*")
        st.text(f"RMS Audio Power: {metrics_left['low']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_left['low']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_left['low']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_left['low']['crest_factor']:.2f}")
    
    with col_pwr2_l:
        st.markdown("*Mid Band*")
        st.text(f"RMS Audio Power: {metrics_left['mid']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_left['mid']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_left['mid']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_left['mid']['crest_factor']:.2f}")
    
    with col_pwr3_l:
        st.markdown("*High Band*")
        st.text(f"RMS Audio Power: {metrics_left['high']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_left['high']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_left['high']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_left['high']['crest_factor']:.2f}")
    
    # Display Right Channel metrics
    st.markdown("**Right Channel**")
    col_pwr1_r, col_pwr2_r, col_pwr3_r = st.columns(3)
    
    with col_pwr1_r:
        st.markdown("*Low Band*")
        st.text(f"RMS Audio Power: {metrics_right['low']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_right['low']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_right['low']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_right['low']['crest_factor']:.2f}")
    
    with col_pwr2_r:
        st.markdown("*Mid Band*")
        st.text(f"RMS Audio Power: {metrics_right['mid']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_right['mid']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_right['mid']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_right['mid']['crest_factor']:.2f}")
    
    with col_pwr3_r:
        st.markdown("*High Band*")
        st.text(f"RMS Audio Power: {metrics_right['high']['rms_power']:.3f} W")
        st.text(f"Peak Current:    {metrics_right['high']['peak_current']:.3f} A")
        st.text(f"Total Energy:    {metrics_right['high']['energy']:.3f} J")
        st.text(f"Crest Factor:    {metrics_right['high']['crest_factor']:.2f}")
    
    # Total power summary (both channels)
    st.markdown("---")
    total_rms_left = metrics_left['low']['rms_power'] + metrics_left['mid']['rms_power'] + metrics_left['high']['rms_power']
    total_rms_right = metrics_right['low']['rms_power'] + metrics_right['mid']['rms_power'] + metrics_right['high']['rms_power']
    total_energy_left = metrics_left['low']['energy'] + metrics_left['mid']['energy'] + metrics_left['high']['energy']
    total_energy_right = metrics_right['low']['energy'] + metrics_right['mid']['energy'] + metrics_right['high']['energy']
    
    col_total_l, col_total_r = st.columns(2)
    with col_total_l:
        st.text(f"Left Total:  RMS Audio Power: {total_rms_left:.3f} W  |  Energy: {total_energy_left:.3f} J")
    with col_total_r:
        st.text(f"Right Total: RMS Audio Power: {total_rms_right:.3f} W  |  Energy: {total_energy_right:.3f} J")
    
    st.text(f"Combined Total: RMS Audio Power: {total_rms_left + total_rms_right:.3f} W  |  Energy: {total_energy_left + total_energy_right:.3f} J")
    
    # -------------------------------------------------------------------------
    # SPL vs Frequency Plot (using calculated power)
    # -------------------------------------------------------------------------
    # Check if we have T/S parameters from the acoustics section
    try:
        # Get calculated RMS power per band (sum of L+R for two drivers per band)
        # Two drivers = double power = +3dB SPL
        power_per_band = {
            'Low': metrics_left['low']['rms_power'] + metrics_right['low']['rms_power'],
            'Mid': metrics_left['mid']['rms_power'] + metrics_right['mid']['rms_power'],
            'High': metrics_left['high']['rms_power'] + metrics_right['high']['rms_power']
        }
        
        st.markdown("---")
        st.markdown(f"**SPL vs Frequency (Calculated Power, 1m, 2 drivers per band)**")
        st.caption(f"Total RMS power (L+R): Low={power_per_band['Low']:.3f}W, Mid={power_per_band['Mid']:.3f}W, High={power_per_band['High']:.3f}W")
        
        # Get crossover responses for SPL plot
        xover_responses_spl = dsp.get_crossover_responses(
            low_mid_freq, mid_high_freq, sample_rate, filter_order, low_band_hpf
        )
        xover_db = {
            'Low': xover_responses_spl['low_db'],
            'Mid': xover_responses_spl['mid_db'],
            'High': xover_responses_spl['high_db']
        }
        xover_freqs = xover_responses_spl['frequencies']
        
        # Calculate SPL response for each band with T/S params, EQ, and crossover
        fig_spl = go.Figure()
        
        colors = {'Low': '#e74c3c', 'Mid': '#2ecc71', 'High': '#3498db'}
        
        for ts_params, band_name in [
            (ts_params_low, "Low"),
            (ts_params_mid, "Mid"),
            (ts_params_high, "High")
        ]:
            if ts_params is not None:
                spl_response = dsp.calculate_spl_response(ts_params)
                
                # Apply all 3 EQ bands to SPL response
                eq_bands = eq_settings[band_name]
                spl_with_eq = spl_response['spl'].copy()
                
                for eq_band in eq_bands:
                    if eq_band['gain'] != 0:
                        eq_response = dsp.get_parametric_eq_response(
                            eq_band['freq'], eq_band['q'], eq_band['gain'], sample_rate
                        )
                        # Interpolate EQ response to match SPL frequencies
                        eq_interp = np.interp(
                            spl_response['frequencies'],
                            eq_response['frequencies'],
                            eq_response['magnitude_db']
                        )
                        spl_with_eq = spl_with_eq + eq_interp
                
                # Apply crossover filter response
                xover_interp = np.interp(
                    spl_response['frequencies'],
                    xover_freqs,
                    xover_db[band_name]
                )
                spl_with_xover = spl_with_eq + xover_interp
                
                # Scale SPL by actual calculated power (SPL increases by 10*log10(power) above 1W reference)
                power_db = 10 * np.log10(power_per_band[band_name] + 1e-10)
                spl_scaled = spl_with_xover + power_db
                
                fig_spl.add_trace(go.Scatter(
                    x=spl_response['frequencies'],
                    y=spl_scaled,
                    mode='lines',
                    line=dict(color=colors[band_name], width=2),
                    name=f'{band_name} ({power_per_band[band_name]:.3f}W)'
                ))
                
                # Add Fs marker
                fig_spl.add_vline(
                    x=ts_params['fs'], 
                    line_dash="dot", 
                    line_color=colors[band_name],
                    opacity=0.5
                )
        
        fig_spl.update_layout(
            xaxis_title="Frequency (Hz)",
            yaxis_title="SPL (dB)",
            xaxis_type="log",
            template="plotly_dark",
            height=300,
            margin=dict(l=50, r=20, t=20, b=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.3)',
            xaxis=dict(range=[np.log10(10), np.log10(20000)]),
            yaxis=dict(range=[40, 120]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_spl, use_container_width=True, key="spl_response")
    
    except NameError:
        # T/S parameters or EQ settings not available (no impedance data loaded)
        pass
    
    # -------------------------------------------------------------------------
    # System Power Analysis
    # -------------------------------------------------------------------------
    st.subheader("System Power Analysis")
    
    total_audio_power = total_rms_left + total_rms_right
    
    col_sys1, col_sys2, col_sys3 = st.columns(3)
    
    with col_sys1:
        dc_efficiency = st.number_input(
            "DC Conversion Efficiency (%)",
            min_value=1.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help="Efficiency of DC-DC converter"
        )
        
        amp_efficiency = st.number_input(
            "Amplifier Efficiency (%)",
            min_value=1.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Efficiency of audio amplifier"
        )
    
    with col_sys2:
        digital_power = st.number_input(
            "Digital Power (W)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Power consumed by DSP, microcontrollers, etc."
        )
        
        amp_standby_power = st.number_input(
            "Amplifier Standby Power (W)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            help="Amplifier quiescent/standby power consumption"
        )
        
        battery_energy = st.number_input(
            "Battery Energy Storage (Wh)",
            min_value=0.1,
            max_value=1000.0,
            value=250.0,
            step=0.1,
            help="Total battery capacity in Watt-hours"
        )
    
    with col_sys3:
        # Calculate battery power
        dc_eff = dc_efficiency / 100.0
        amp_eff = amp_efficiency / 100.0
        
        non_audio_power = digital_power + amp_standby_power
        audio_power_from_battery = total_audio_power / amp_eff / dc_eff
        total_battery_power = audio_power_from_battery + non_audio_power
        
        # Calculate runtime
        if total_battery_power > 0:
            runtime_hours = battery_energy / total_battery_power
            runtime_minutes = runtime_hours * 60
        else:
            runtime_hours = float('inf')
            runtime_minutes = float('inf')
        
        st.markdown("**Calculated Values**")
        st.text(f"Battery Power Required: {total_battery_power:.3f} W")
        st.text(f"Runtime: {runtime_hours:.2f} hrs ({runtime_minutes:.1f} min)")
    
    # -------------------------------------------------------------------------
    # Energy Distribution Pie Chart
    # -------------------------------------------------------------------------
    st.subheader("Energy Distribution")
    
    # Calculate power for each category
    low_power = (metrics_left['low']['rms_power'] + metrics_right['low']['rms_power'])
    mid_power = (metrics_left['mid']['rms_power'] + metrics_right['mid']['rms_power'])
    high_power = (metrics_left['high']['rms_power'] + metrics_right['high']['rms_power'])
    
    # Amplifier losses: power into amp - audio power out
    amp_input_power = total_audio_power / amp_eff
    amp_losses = amp_input_power - total_audio_power
    
    # DC converter losses: power from battery for audio - power into amp
    dc_input_power = amp_input_power / dc_eff
    dc_losses = dc_input_power - amp_input_power
    
    # Build pie chart data
    labels = ['Low Band', 'Mid Band', 'High Band', 'Digital', 'Amp Standby', 'DC Converter Losses', 'Amplifier Losses']
    values = [low_power, mid_power, high_power, digital_power, amp_standby_power, dc_losses, amp_losses]
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#8e44ad', '#f39c12', '#e67e22']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        hole=0.3
    )])
    fig_pie.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True, key="energy_pie")
    
    # -------------------------------------------------------------------------
    # Download Processed Audio
    # -------------------------------------------------------------------------
    st.subheader("Export")
    
    stereo_export = np.column_stack((audio_left, audio_right))
    export_buffer = BytesIO()
    sf.write(export_buffer, stereo_export, sample_rate, format='WAV')
    
    st.download_button(
        label="Download Processed Audio",
        data=export_buffer.getvalue(),
        file_name=f"processed_{st.session_state.filename.rsplit('.', 1)[0]}.wav",
        mime="audio/wav"
    )

else:
    st.info("Upload an audio file to get started")
