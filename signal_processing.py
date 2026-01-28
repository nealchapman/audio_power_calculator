"""
Audio Signal Processing Module
Clean, modular functions for audio analysis and processing
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d


# =============================================================================
# FFT Analysis
# =============================================================================

def compute_fft(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Compute FFT of audio data.
    
    Args:
        audio_data: Time-domain audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        dict with 'frequencies', 'magnitudes', 'phases'
    """
    n = len(audio_data)
    
    # Apply Hanning window
    windowed = audio_data * np.hanning(n)
    
    # Compute FFT
    fft_result = fft(windowed)
    frequencies = fftfreq(n, 1/sample_rate)
    
    # Take positive frequencies only
    positive_mask = frequencies >= 0
    frequencies = frequencies[positive_mask]
    fft_result = fft_result[positive_mask]
    
    magnitudes = np.abs(fft_result) / n
    phases = np.angle(fft_result)
    
    return {
        'frequencies': frequencies,
        'magnitudes': magnitudes,
        'phases': phases
    }


def compute_power_spectrum(magnitudes: np.ndarray) -> np.ndarray:
    """
    Compute power spectrum in dB.
    
    Args:
        magnitudes: FFT magnitudes
        
    Returns:
        Power in dB
    """
    return 20 * np.log10(magnitudes + 1e-10)


# =============================================================================
# Time-Domain Analysis
# =============================================================================

def calculate_rms(audio_data: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) level."""
    return np.sqrt(np.mean(audio_data ** 2))


def calculate_peak(audio_data: np.ndarray) -> float:
    """Calculate peak absolute amplitude."""
    return np.max(np.abs(audio_data))


def calculate_crest_factor(audio_data: np.ndarray) -> float:
    """Calculate crest factor (peak / RMS)."""
    rms = calculate_rms(audio_data)
    if rms == 0:
        return 0
    return calculate_peak(audio_data) / rms


def calculate_zero_crossing_rate(audio_data: np.ndarray) -> float:
    """Calculate zero-crossing rate (crossings per sample)."""
    signs = np.sign(audio_data)
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return crossings / len(audio_data)


def truncate_to_zero_crossings(audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Truncate audio to start and end near zero crossings.
    
    Finds the first and last points where the absolute value is below
    the threshold, helping avoid clicks at start/end of playback.
    
    Args:
        audio_data: Audio samples
        threshold: Maximum absolute value to consider "near zero"
        
    Returns:
        Truncated audio array
    """
    if len(audio_data) == 0:
        return audio_data
    
    # Find indices where signal is near zero
    near_zero = np.abs(audio_data) < threshold
    
    # Find first near-zero point
    start_indices = np.where(near_zero)[0]
    if len(start_indices) > 0:
        start_idx = start_indices[0]
    else:
        start_idx = 0
    
    # Find last near-zero point
    if len(start_indices) > 0:
        end_idx = start_indices[-1] + 1  # +1 to include the point
    else:
        end_idx = len(audio_data)
    
    # Ensure we have a valid range
    if start_idx >= end_idx:
        return audio_data
    
    return audio_data[start_idx:end_idx]


# =============================================================================
# Filtering
# =============================================================================

def low_pass_filter(audio_data: np.ndarray, cutoff_freq: float, 
                    sample_rate: int, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter.
    
    Args:
        audio_data: Audio samples
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered audio
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, audio_data)


def high_pass_filter(audio_data: np.ndarray, cutoff_freq: float,
                     sample_rate: int, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter.
    
    Args:
        audio_data: Audio samples
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered audio
    """
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    return signal.filtfilt(b, a, audio_data)


def band_pass_filter(audio_data: np.ndarray, low_freq: float, high_freq: float,
                     sample_rate: int, order: int = 4) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.
    
    Args:
        audio_data: Audio samples
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered audio
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, audio_data)


# =============================================================================
# 3-Way Crossover
# =============================================================================

def three_way_crossover(audio_data: np.ndarray, low_mid_freq: float, 
                        mid_high_freq: float, sample_rate: int,
                        order: int = 4, low_band_hpf: float = None) -> dict:
    """
    Apply a 3-way crossover to split audio into low, mid, and high bands.
    
    Uses Linkwitz-Riley style crossover (cascaded Butterworth filters)
    for flat summed response.
    
    Args:
        audio_data: Audio samples
        low_mid_freq: Crossover frequency between low and mid bands (Hz)
        mid_high_freq: Crossover frequency between mid and high bands (Hz)
        sample_rate: Sample rate in Hz
        order: Filter order (slope = order * 6 dB/octave)
        low_band_hpf: Optional additional highpass filter on low band (Hz), 24dB/oct
        
    Returns:
        dict with 'low', 'mid', 'high' band arrays
    """
    nyquist = sample_rate / 2
    
    # Normalize frequencies
    low_mid_norm = low_mid_freq / nyquist
    mid_high_norm = mid_high_freq / nyquist
    
    # Design filters
    b_low, a_low = signal.butter(order, low_mid_norm, btype='low')
    b_high, a_high = signal.butter(order, mid_high_norm, btype='high')
    b_mid, a_mid = signal.butter(order, [low_mid_norm, mid_high_norm], btype='band')
    
    # Calculate padding length based on filter order and lowest frequency
    # Longer padding for lower frequencies and higher orders
    pad_samples = int(sample_rate / low_mid_freq * order * 6)
    pad_samples = min(pad_samples, len(audio_data) // 2)  # Don't exceed half signal length
    
    # Apply filters with extended padding to reduce edge transients
    low_band = signal.filtfilt(b_low, a_low, audio_data, padlen=pad_samples)
    mid_band = signal.filtfilt(b_mid, a_mid, audio_data, padlen=pad_samples)
    high_band = signal.filtfilt(b_high, a_high, audio_data, padlen=pad_samples)
    
    # Apply optional additional highpass filter to low band (24dB/oct = 2nd order * 2 from filtfilt)
    if low_band_hpf is not None and low_band_hpf > 0:
        hpf_norm = low_band_hpf / nyquist
        # Ensure we don't exceed Nyquist
        if hpf_norm < 1.0:
            b_hpf, a_hpf = signal.butter(2, hpf_norm, btype='high')  # 2nd order, filtfilt doubles to 24dB/oct
            hpf_pad = int(sample_rate / low_band_hpf * 4 * 6)
            hpf_pad = min(hpf_pad, len(audio_data) // 2)
            low_band = signal.filtfilt(b_hpf, a_hpf, low_band, padlen=hpf_pad)
    
    return {
        'low': low_band,
        'mid': mid_band,
        'high': high_band
    }


def get_filter_response(cutoff_freq: float, sample_rate: int, 
                        filter_type: str, order: int = 4,
                        num_points: int = 500) -> dict:
    """
    Compute frequency response of a filter for visualization.
    
    Args:
        cutoff_freq: Cutoff frequency in Hz (or list of [low, high] for bandpass)
        sample_rate: Sample rate in Hz
        filter_type: 'low', 'high', or 'band'
        order: Filter order
        num_points: Number of frequency points
        
    Returns:
        dict with 'frequencies' and 'magnitude_db'
    """
    nyquist = sample_rate / 2
    
    if filter_type == 'band':
        normalized = [f / nyquist for f in cutoff_freq]
    else:
        normalized = cutoff_freq / nyquist
    
    b, a = signal.butter(order, normalized, btype=filter_type)
    
    # Use logarithmically spaced frequencies for better low-frequency resolution
    frequencies = np.logspace(np.log10(1), np.log10(nyquist * 0.99), num_points)
    w = frequencies * np.pi / nyquist
    _, h = signal.freqz(b, a, worN=w)
    
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
    
    return {
        'frequencies': frequencies,
        'magnitude_db': magnitude_db
    }


def get_crossover_responses(low_mid_freq: float, mid_high_freq: float,
                            sample_rate: int, order: int = 4,
                            low_band_hpf: float = None) -> dict:
    """
    Get frequency responses for all bands in a 3-way crossover.
    
    Args:
        low_mid_freq: Low-to-mid crossover frequency (Hz)
        mid_high_freq: Mid-to-high crossover frequency (Hz)
        sample_rate: Sample rate in Hz
        order: Filter order
        low_band_hpf: Optional additional highpass filter on low band (Hz), 24dB/oct
        
    Returns:
        dict with 'frequencies', 'low_db', 'mid_db', 'high_db'
    """
    low_resp = get_filter_response(low_mid_freq, sample_rate, 'low', order)
    mid_resp = get_filter_response([low_mid_freq, mid_high_freq], sample_rate, 'band', order)
    high_resp = get_filter_response(mid_high_freq, sample_rate, 'high', order)
    
    # Apply additional HPF to low band response if specified
    # Use order 4 to match filtfilt doubling (2nd order * 2 = 4th order = 24dB/oct)
    low_db = np.array(low_resp['magnitude_db'])
    if low_band_hpf is not None and low_band_hpf > 0:
        hpf_resp = get_filter_response(low_band_hpf, sample_rate, 'high', 4)  # 24dB/oct effective
        low_db = low_db + np.array(hpf_resp['magnitude_db'])  # Add dB (multiply linear)
    
    return {
        'frequencies': low_resp['frequencies'],
        'low_db': low_db,
        'mid_db': mid_resp['magnitude_db'],
        'high_db': high_resp['magnitude_db']
    }


# =============================================================================
# Parametric EQ
# =============================================================================

def parametric_eq(audio_data: np.ndarray, sample_rate: int, 
                  freq: float, q: float, gain_db: float) -> np.ndarray:
    """
    Apply a parametric EQ (peaking filter) to audio data.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        freq: Center frequency in Hz
        q: Q factor (bandwidth control)
        gain_db: Gain in dB (positive = boost, negative = cut)
        
    Returns:
        Filtered audio data
    """
    if gain_db == 0 or freq <= 0 or q <= 0:
        return audio_data
    
    # Calculate filter coefficients (peaking EQ / parametric)
    A = 10 ** (gain_db / 40)  # amplitude
    omega = 2 * np.pi * freq / sample_rate
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    alpha = sin_omega / (2 * q)
    
    # Peaking EQ coefficients
    b0 = 1 + alpha * A
    b1 = -2 * cos_omega
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_omega
    a2 = 1 - alpha / A
    
    # Normalize coefficients
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    
    # Apply filter (zero-phase filtering)
    pad_samples = min(int(sample_rate / freq * 4), len(audio_data) // 2)
    filtered = signal.filtfilt(b, a, audio_data, padlen=pad_samples)
    
    return filtered


def get_parametric_eq_response(freq: float, q: float, gain_db: float,
                                sample_rate: int, num_points: int = 500) -> dict:
    """
    Get the frequency response of a parametric EQ.
    
    Args:
        freq: Center frequency in Hz
        q: Q factor
        gain_db: Gain in dB
        sample_rate: Sample rate in Hz
        num_points: Number of frequency points
        
    Returns:
        dict with 'frequencies' and 'magnitude_db'
    """
    if gain_db == 0 or freq <= 0 or q <= 0:
        frequencies = np.logspace(np.log10(20), np.log10(20000), num_points)
        return {
            'frequencies': frequencies,
            'magnitude_db': np.zeros(num_points)
        }
    
    # Calculate filter coefficients
    A = 10 ** (gain_db / 40)
    omega = 2 * np.pi * freq / sample_rate
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    alpha = sin_omega / (2 * q)
    
    b0 = 1 + alpha * A
    b1 = -2 * cos_omega
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_omega
    a2 = 1 - alpha / A
    
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1, a1/a0, a2/a0])
    
    # Get frequency response (log spaced)
    frequencies = np.logspace(np.log10(20), np.log10(sample_rate * 0.45), num_points)
    w = frequencies * np.pi / (sample_rate / 2)
    _, h = signal.freqz(b, a, worN=w)
    
    # Account for filtfilt doubling
    magnitude_db = 2 * 20 * np.log10(np.abs(h) + 1e-10)
    
    return {
        'frequencies': frequencies,
        'magnitude_db': magnitude_db
    }


# =============================================================================
# Windowing Functions
# =============================================================================

def apply_window(audio_data: np.ndarray, window_type: str = 'hanning') -> np.ndarray:
    """
    Apply a window function to audio data.
    
    Args:
        audio_data: Audio samples
        window_type: 'hanning', 'hamming', 'blackman', or 'rectangular'
        
    Returns:
        Windowed audio
    """
    n = len(audio_data)
    
    windows = {
        'hanning': np.hanning(n),
        'hamming': np.hamming(n),
        'blackman': np.blackman(n),
        'rectangular': np.ones(n)
    }
    
    window = windows.get(window_type, np.hanning(n))
    return audio_data * window


# =============================================================================
# Utility Functions
# =============================================================================

def downsample_for_display(audio_data: np.ndarray, target_length: int) -> np.ndarray:
    """
    Downsample audio for visualization using max absolute value per bin.
    
    Args:
        audio_data: Audio samples
        target_length: Target number of samples
        
    Returns:
        Downsampled audio
    """
    if len(audio_data) <= target_length:
        return audio_data
    
    ratio = len(audio_data) / target_length
    downsampled = np.zeros(target_length)
    
    for i in range(target_length):
        start = int(i * ratio)
        end = int((i + 1) * ratio)
        chunk = audio_data[start:end]
        # Use value with max absolute for visualization
        idx = np.argmax(np.abs(chunk))
        downsampled[i] = chunk[idx]
    
    return downsampled


def get_audio_stats(audio_data: np.ndarray, sample_rate: int) -> dict:
    """
    Get comprehensive audio statistics.
    
    Args:
        audio_data: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        dict with audio statistics
    """
    return {
        'duration': len(audio_data) / sample_rate,
        'sample_rate': sample_rate,
        'samples': len(audio_data),
        'rms': calculate_rms(audio_data),
        'peak': calculate_peak(audio_data),
        'crest_factor': calculate_crest_factor(audio_data),
        'zero_crossing_rate': calculate_zero_crossing_rate(audio_data)
    }


# =============================================================================
# Impedance Functions
# =============================================================================

def load_impedance_csv(file_content: bytes) -> dict:
    """
    Load impedance data from a CSV file.
    
    Expected CSV format:
        frequency,impedance,phase
        10,12.5,45
        ...
    
    Args:
        file_content: Raw bytes from uploaded CSV file
        
    Returns:
        dict with 'frequencies', 'impedance', 'phase' arrays
    """
    from io import StringIO
    
    # Decode and read CSV
    content_str = file_content.decode('utf-8')
    df = pd.read_csv(StringIO(content_str))
    
    # Normalize column names (lowercase, strip whitespace)
    df.columns = df.columns.str.lower().str.strip()
    
    return {
        'frequencies': df['frequency'].values.astype(float),
        'impedance': df['impedance'].values.astype(float),
        'phase': df['phase'].values.astype(float)
    }


def interpolate_impedance(impedance_data: dict, target_frequencies: np.ndarray) -> dict:
    """
    Interpolate impedance data to target frequencies.
    
    Args:
        impedance_data: dict with 'frequencies', 'impedance', 'phase'
        target_frequencies: Array of frequencies to interpolate to
        
    Returns:
        dict with interpolated 'impedance' and 'phase' arrays
    """
    # Create interpolation functions (log scale for frequency)
    freq_log = np.log10(impedance_data['frequencies'] + 1e-10)
    target_log = np.log10(np.maximum(target_frequencies, 1e-10))
    
    # Clamp target frequencies to data range
    min_freq = impedance_data['frequencies'].min()
    max_freq = impedance_data['frequencies'].max()
    
    interp_impedance = interp1d(
        freq_log, impedance_data['impedance'],
        kind='linear', bounds_error=False,
        fill_value=(impedance_data['impedance'][0], impedance_data['impedance'][-1])
    )
    
    interp_phase = interp1d(
        freq_log, impedance_data['phase'],
        kind='linear', bounds_error=False,
        fill_value=(impedance_data['phase'][0], impedance_data['phase'][-1])
    )
    
    return {
        'impedance': interp_impedance(target_log),
        'phase': interp_phase(target_log)
    }


def compute_power_with_impedance(voltage_signal: np.ndarray, sample_rate: int,
                                  impedance_data: dict) -> dict:
    """
    Compute power metrics using frequency-dependent complex impedance.
    
    Converts voltage to frequency domain, applies complex impedance,
    then computes current and power.
    
    Args:
        voltage_signal: Time-domain voltage signal
        sample_rate: Sample rate in Hz
        impedance_data: dict with 'frequencies', 'impedance', 'phase'
        
    Returns:
        dict with power metrics and signals
    """
    n = len(voltage_signal)
    
    # Compute FFT of voltage
    V_fft = fft(voltage_signal)
    freqs = fftfreq(n, 1/sample_rate)
    
    # Get positive frequencies for interpolation
    pos_mask = freqs >= 0
    pos_freqs = freqs[pos_mask]
    
    # Interpolate impedance to FFT frequencies
    interp_data = interpolate_impedance(impedance_data, np.abs(pos_freqs))
    Z_mag = interp_data['impedance']
    Z_phase_deg = interp_data['phase']
    Z_phase_rad = np.deg2rad(Z_phase_deg)
    
    # Create complex impedance (for positive frequencies)
    Z_complex_pos = Z_mag * np.exp(1j * Z_phase_rad)
    
    # Create full complex impedance array (handle negative frequencies)
    Z_complex = np.zeros(n, dtype=complex)
    Z_complex[pos_mask] = Z_complex_pos
    # Negative frequencies get conjugate
    neg_mask = freqs < 0
    neg_freqs_abs = np.abs(freqs[neg_mask])
    interp_neg = interpolate_impedance(impedance_data, neg_freqs_abs)
    Z_mag_neg = interp_neg['impedance']
    Z_phase_neg_rad = np.deg2rad(interp_neg['phase'])
    Z_complex[neg_mask] = Z_mag_neg * np.exp(-1j * Z_phase_neg_rad)
    
    # Avoid division by zero at DC
    Z_complex[0] = max(Z_complex[1].real, 0.1) if n > 1 else 1.0
    
    # Compute current in frequency domain: I = V / Z
    I_fft = V_fft / Z_complex
    
    # Convert current back to time domain
    current_signal = np.real(ifft(I_fft))
    
    # Compute instantaneous power: P = V * I
    power_signal = voltage_signal * current_signal
    
    # Compute metrics
    rms_voltage = calculate_rms(voltage_signal)
    rms_current = calculate_rms(current_signal)
    rms_power = np.mean(power_signal)  # Average power
    peak_current = calculate_peak(current_signal)
    total_energy = np.sum(np.abs(power_signal)) / sample_rate
    crest_factor = calculate_crest_factor(voltage_signal)
    
    return {
        'rms_power': rms_power,
        'peak_current': peak_current,
        'energy': total_energy,
        'crest_factor': crest_factor,
        'rms_voltage': rms_voltage,
        'rms_current': rms_current,
        'current_signal': current_signal,
        'power_signal': power_signal
    }


def compute_power_with_resistance(voltage_signal: np.ndarray, resistance: float,
                                   sample_rate: int) -> dict:
    """
    Compute power metrics using simple resistance (no frequency dependence).
    
    Args:
        voltage_signal: Time-domain voltage signal
        resistance: Resistance in Ohms
        sample_rate: Sample rate in Hz
        
    Returns:
        dict with power metrics
    """
    current_signal = voltage_signal / resistance
    power_signal = voltage_signal ** 2 / resistance
    
    return {
        'rms_power': np.mean(power_signal),
        'peak_current': calculate_peak(current_signal),
        'energy': np.sum(power_signal) / sample_rate,
        'crest_factor': calculate_crest_factor(voltage_signal),
        'rms_voltage': calculate_rms(voltage_signal),
        'rms_current': calculate_rms(current_signal),
        'current_signal': current_signal,
        'power_signal': power_signal
    }


def compute_power_convolution(voltage_signal: np.ndarray, sample_rate: int,
                               impedance_data: dict) -> dict:
    """
    Compute power by convolving voltage with the impulse response of impedance.
    
    The impedance Z(f) is converted to an impulse response h(t) via IFFT,
    then current is computed by convolving voltage with h(t).
    This is equivalent to I(f) = V(f) / Z(f), but done in time domain.
    
    Actually computes admittance Y(f) = 1/Z(f), then IFFT to get impulse
    response, since I = V * Y in frequency domain means i(t) = v(t) * y(t)
    in time domain (convolution).
    
    Args:
        voltage_signal: Time-domain voltage signal
        sample_rate: Sample rate in Hz
        impedance_data: dict with 'frequencies', 'impedance', 'phase'
        
    Returns:
        dict with power metrics and signals
    """
    n = len(voltage_signal)
    
    # Build frequency array for the FFT
    freqs = fftfreq(n, 1/sample_rate)
    
    # Get positive frequencies for interpolation
    pos_mask = freqs >= 0
    pos_freqs = np.abs(freqs[pos_mask])
    
    # Interpolate impedance to FFT frequencies
    interp_data = interpolate_impedance(impedance_data, pos_freqs)
    Z_mag = interp_data['impedance']
    Z_phase_deg = interp_data['phase']
    Z_phase_rad = np.deg2rad(Z_phase_deg)
    
    # Create complex impedance for positive frequencies
    Z_complex_pos = Z_mag * np.exp(1j * Z_phase_rad)
    
    # Compute admittance Y = 1/Z for positive frequencies
    Y_complex_pos = 1.0 / Z_complex_pos
    
    # Build full admittance spectrum (handle negative frequencies with conjugate)
    Y_spectrum = np.zeros(n, dtype=complex)
    Y_spectrum[pos_mask] = Y_complex_pos
    
    neg_mask = freqs < 0
    neg_freqs_abs = np.abs(freqs[neg_mask])
    interp_neg = interpolate_impedance(impedance_data, neg_freqs_abs)
    Z_mag_neg = interp_neg['impedance']
    Z_phase_neg_rad = np.deg2rad(interp_neg['phase'])
    Z_complex_neg = Z_mag_neg * np.exp(-1j * Z_phase_neg_rad)  # Conjugate for negative freqs
    Y_spectrum[neg_mask] = 1.0 / Z_complex_neg
    
    # Handle DC component (avoid division by zero)
    if n > 1:
        Y_spectrum[0] = 1.0 / max(Z_mag[0], 0.1)
    
    # Compute impulse response of admittance via IFFT
    # Use fftshift to make the impulse response causal (centered, then we take the causal part)
    admittance_impulse_raw = ifft(Y_spectrum)
    admittance_impulse_shifted = np.fft.fftshift(admittance_impulse_raw)
    
    # Take the real part and keep only the causal portion (second half)
    # This represents t >= 0
    admittance_impulse = np.real(admittance_impulse_shifted[n//2:])
    
    # Convolve voltage with causal admittance impulse response to get current
    # Use 'full' mode then truncate to maintain causality
    current_full = np.convolve(voltage_signal, admittance_impulse, mode='full')
    current_signal = current_full[:n]  # Keep same length as input
    
    # Scale by sample spacing (dt) for proper convolution normalization
    dt = 1.0 / sample_rate
    current_signal = current_signal * dt
    
    # Compute instantaneous power: P = V * I
    power_signal = voltage_signal * current_signal
    
    # Compute metrics
    rms_voltage = calculate_rms(voltage_signal)
    rms_current = calculate_rms(current_signal)
    rms_power = np.mean(power_signal)
    peak_current = calculate_peak(current_signal)
    total_energy = np.sum(np.abs(power_signal)) / sample_rate
    crest_factor = calculate_crest_factor(voltage_signal)
    
    return {
        'rms_power': rms_power,
        'peak_current': peak_current,
        'energy': total_energy,
        'crest_factor': crest_factor,
        'rms_voltage': rms_voltage,
        'rms_current': rms_current,
        'current_signal': current_signal,
        'power_signal': power_signal,
        'admittance_impulse': admittance_impulse
    }


# =============================================================================
# Thiele/Small Parameter Calculations
# =============================================================================

def calculate_thiele_small_parameters(impedance_data: dict, sd: float, mmd: float) -> dict:
    """
    Calculate Thiele/Small parameters from impedance data.
    
    Uses the impedance magnitude and phase to extract motor and mechanical
    parameters. Requires user-provided diaphragm area (Sd) and moving mass (Mmd).
    
    Args:
        impedance_data: dict with 'frequencies', 'impedance', 'phase' arrays
        sd: Effective diaphragm area in m² (e.g., 0.0012 for a small driver)
        mmd: Diaphragm/voice coil mass in kg (e.g., 0.005 for 5 grams)
        
    Returns:
        dict with all Thiele/Small parameters
    """
    frequencies = np.array(impedance_data['frequencies'])
    impedance = np.array(impedance_data['impedance'])
    phase = np.array(impedance_data['phase'])
    
    # Find resonant frequency (Fs) - peak impedance
    fs_idx = np.argmax(impedance)
    fs = frequencies[fs_idx]
    z_max = impedance[fs_idx]
    
    # Find DC resistance (Re) - minimum impedance at low frequency
    # Use the lowest frequency points where phase is near zero
    low_freq_mask = frequencies < fs * 0.5
    if np.any(low_freq_mask):
        re = np.min(impedance[low_freq_mask])
    else:
        re = impedance[0]
    
    # Find -3dB points (f1 and f2) for Q calculation
    # These are where |Z| = sqrt(Re * Zmax)
    z_3db = np.sqrt(re * z_max)
    
    # Find f1 (below Fs)
    below_fs = frequencies < fs
    if np.any(below_fs):
        below_fs_idx = np.where(below_fs)[0]
        # Find where impedance crosses z_3db from below
        crossings_f1 = []
        for i in range(len(below_fs_idx) - 1):
            idx = below_fs_idx[i]
            if impedance[idx] <= z_3db <= impedance[idx + 1]:
                # Linear interpolation
                t = (z_3db - impedance[idx]) / (impedance[idx + 1] - impedance[idx] + 1e-10)
                f1 = frequencies[idx] + t * (frequencies[idx + 1] - frequencies[idx])
                crossings_f1.append(f1)
        f1 = crossings_f1[-1] if crossings_f1 else fs * 0.7
    else:
        f1 = fs * 0.7
    
    # Find f2 (above Fs)
    above_fs = frequencies > fs
    if np.any(above_fs):
        above_fs_idx = np.where(above_fs)[0]
        # Find where impedance crosses z_3db from above
        crossings_f2 = []
        for i in range(len(above_fs_idx) - 1):
            idx = above_fs_idx[i]
            if impedance[idx] >= z_3db >= impedance[idx + 1]:
                # Linear interpolation
                t = (impedance[idx] - z_3db) / (impedance[idx] - impedance[idx + 1] + 1e-10)
                f2 = frequencies[idx] + t * (frequencies[idx + 1] - frequencies[idx])
                crossings_f2.append(f2)
        f2 = crossings_f2[0] if crossings_f2 else fs * 1.4
    else:
        f2 = fs * 1.4
    
    # Calculate Q values
    qms_plus_qes = fs / (f2 - f1) if f2 > f1 else 10.0  # Qts from bandwidth
    qts = qms_plus_qes
    
    # R0 is the ratio of Zmax to Re
    r0 = z_max / re
    
    # Qes and Qms from the impedance ratio
    # Qms = Qts * sqrt(R0)
    # Qes = Qts * sqrt(R0) / (R0 - 1)
    qms = qts * np.sqrt(r0)
    qes = qms / (r0 - 1) if r0 > 1 else qms
    
    # Angular resonant frequency
    omega_s = 2 * np.pi * fs
    
    # Mechanical compliance: Cms = 1 / (omega_s^2 * Mms)
    # First we need total moving mass Mms = Mmd + Mair
    # Air load mass approximation: Mair ≈ 2 * rho_air * Sd^1.5 / sqrt(pi)
    rho_air = 1.18  # kg/m³ at 25°C
    mair = 2 * rho_air * (sd ** 1.5) / np.sqrt(np.pi)
    mms = mmd + mair
    
    # Mechanical compliance
    cms = 1.0 / (omega_s ** 2 * mms)
    
    # Mechanical resistance: Rms = omega_s * Mms / Qms
    rms = omega_s * mms / qms if qms > 0 else 0
    
    # BL product (force factor): BL = sqrt(Re * Rms * (Qms/Qes))
    # From Qes = Re / (BL^2 / Rms) = Re * Rms / BL^2
    # So BL^2 = Re * Rms / Qes
    bl_squared = re * omega_s * mms / qes if qes > 0 else 0
    bl = np.sqrt(bl_squared) if bl_squared > 0 else 0
    
    # Equivalent compliance volume: Vas = rho * c^2 * Cms * Sd^2
    c_air = 343  # m/s speed of sound at 20°C
    vas = rho_air * (c_air ** 2) * cms * (sd ** 2)
    vas_liters = vas * 1000  # Convert to liters
    
    # Reference efficiency: η0 = (rho * BL^2 * Sd^2) / (2 * pi * c * Mms^2 * Re)
    eta0 = (rho_air * bl_squared * sd ** 2) / (2 * np.pi * c_air * mms ** 2 * re)
    eta0_percent = eta0 * 100
    
    # Sensitivity: SPL = 112.1 + 10*log10(η0) dB (1W/1m)
    spl_1w1m = 112.1 + 10 * np.log10(eta0 + 1e-10)
    
    # Electrical equivalent inductance and capacitance
    # Les (electrical equivalent of Mms): Les = BL^2 / Rms = BL^2 * Qms / (omega_s * Mms)
    les = bl_squared * qms / (omega_s * mms) if omega_s * mms > 0 else 0
    
    # Ces (electrical equivalent of Cms): Ces = Mms / BL^2
    ces = mms / bl_squared if bl_squared > 0 else 0
    
    # Res (electrical equivalent of Rms): Res = BL^2 / Rms
    res = bl_squared / rms if rms > 0 else 0
    
    # Xmax estimate (if not provided) - rough estimate from Vas and Sd
    # xmax = Vas / (Sd * some_factor) - this is very rough
    xmax_est = vas / (sd * 10) if sd > 0 else 0  # Very rough estimate
    
    # Vd (displacement volume) = Sd * Xmax
    vd_est = sd * xmax_est * 1e6  # in cm³
    
    return {
        # Fundamental parameters
        'fs': fs,                    # Resonant frequency (Hz)
        're': re,                    # DC resistance (Ohms)
        'qms': qms,                  # Mechanical Q
        'qes': qes,                  # Electrical Q
        'qts': qts,                  # Total Q
        
        # Mass and compliance
        'mms': mms * 1000,           # Total moving mass (grams)
        'mmd': mmd * 1000,           # Diaphragm mass (grams)
        'mair': mair * 1000,         # Air load mass (grams)
        'cms': cms * 1000,           # Mechanical compliance (mm/N)
        
        # Motor parameters
        'bl': bl,                    # Force factor (Tm or N/A)
        'rms': rms,                  # Mechanical resistance (kg/s)
        
        # Volume and area
        'sd': sd * 1e4,              # Effective area (cm²)
        'vas': vas_liters,           # Equivalent volume (liters)
        
        # Impedance characteristics
        'z_max': z_max,              # Peak impedance (Ohms)
        'r0': r0,                    # Impedance ratio
        'f1': f1,                    # Lower -3dB frequency (Hz)
        'f2': f2,                    # Upper -3dB frequency (Hz)
        
        # Efficiency
        'eta0': eta0_percent,        # Reference efficiency (%)
        'spl_1w1m': spl_1w1m,        # Sensitivity (dB @ 1W/1m)
        
        # Electrical equivalents
        'les': les * 1000,           # Electrical inductance (mH)
        'ces': ces * 1e6,            # Electrical capacitance (µF)
        'res': res,                  # Electrical resistance (Ohms)
    }


def calculate_spl_response(ts_params: dict, freq_min: float = 10, 
                           freq_max: float = 20000, num_points: int = 500) -> dict:
    """
    Calculate SPL frequency response from Thiele/Small parameters.
    
    Models the speaker as a 2nd order high-pass system (infinite baffle).
    
    Args:
        ts_params: Dictionary of T/S parameters from calculate_thiele_small_parameters
        freq_min: Minimum frequency (Hz)
        freq_max: Maximum frequency (Hz)
        num_points: Number of frequency points
        
    Returns:
        dict with 'frequencies' and 'spl' arrays
    """
    # Extract parameters (convert units back)
    fs = ts_params['fs']
    qts = ts_params['qts']
    spl_ref = ts_params['spl_1w1m']
    
    # Generate frequency array (log spaced)
    frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_points)
    
    # Angular frequencies
    omega = 2 * np.pi * frequencies
    omega_s = 2 * np.pi * fs
    
    # 2nd order high-pass transfer function
    # H(s) = s² / (s² + (ωs/Qts)*s + ωs²)
    # For s = jω:
    # H(jω) = -ω² / (-ω² + j*(ωs/Qts)*ω + ωs²)
    #       = -ω² / (ωs² - ω² + j*(ωs*ω/Qts))
    
    s = 1j * omega
    s_squared = -omega**2  # (jω)² = -ω²
    
    # Numerator: s²
    numerator = s_squared
    
    # Denominator: s² + (ωs/Qts)*s + ωs²
    denominator = s_squared + (omega_s / qts) * s + omega_s**2
    
    # Transfer function magnitude
    h = numerator / denominator
    h_mag = np.abs(h)
    
    # Convert to dB relative to reference sensitivity
    spl = spl_ref + 20 * np.log10(h_mag + 1e-10)
    
    return {
        'frequencies': frequencies,
        'spl': spl
    }
