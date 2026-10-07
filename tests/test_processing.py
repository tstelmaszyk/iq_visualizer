import numpy as np
import pytest

from iq_visualizer.processing import compute_spectrum, time_axis


def complex_tone(frequency, sample_rate, n_samples):
    times = np.arange(n_samples) / sample_rate
    return np.exp(2j * np.pi * frequency * times)


def test_time_axis_in_seconds_with_sample_rate():
    np.testing.assert_allclose(time_axis(4, 2.0), [0.0, 0.5, 1.0, 1.5])


def test_time_axis_in_samples_without_sample_rate():
    np.testing.assert_array_equal(time_axis(3), [0, 1, 2])


@pytest.mark.parametrize("window", ["hann", "rectangular"])
def test_tone_peak_is_at_its_frequency_and_0_db(window):
    iq = complex_tone(100.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           window=window, scale="db")
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(100.0)
    assert values[peak] == pytest.approx(0.0, abs=1e-9)


def test_tone_peak_is_1_in_linear_scale():
    iq = complex_tone(-250.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           scale="linear")
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(-250.0)
    assert values[peak] == pytest.approx(1.0)


def test_frequencies_are_centered_and_sorted():
    frequencies, _ = compute_spectrum(np.ones(8), sample_rate=8.0)
    np.testing.assert_allclose(frequencies, np.arange(-4, 4))


def test_normalized_frequencies_without_sample_rate():
    frequencies, _ = compute_spectrum(np.ones(8))
    assert frequencies[0] == pytest.approx(-0.5)
    assert frequencies[-1] < 0.5


def test_nfft_smaller_than_signal_truncates():
    frequencies, values = compute_spectrum(np.ones(100), nfft=16)
    assert frequencies.size == values.size == 16


def test_nfft_larger_than_signal_zero_pads():
    iq = complex_tone(100.0, 1000.0, 1000)
    frequencies, values = compute_spectrum(iq, sample_rate=1000.0,
                                           nfft=4000, scale="linear")
    assert frequencies.size == 4000
    peak = np.argmax(values)
    assert frequencies[peak] == pytest.approx(100.0)
    assert values[peak] == pytest.approx(1.0)


def test_db_scale_has_no_infinite_values_on_silence():
    _, values = compute_spectrum(np.zeros(16, dtype=complex),
                                 window="rectangular")
    assert np.all(np.isfinite(values))


def test_single_sample_signal_is_supported():
    frequencies, values = compute_spectrum(np.array([1 + 1j]))
    assert frequencies.size == values.size == 1


@pytest.mark.parametrize("kwargs, message", [
    ({"window": "blackman"}, "Unknown window"),
    ({"scale": "power"}, "Unknown scale"),
])
def test_unknown_option_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compute_spectrum(np.ones(8), **kwargs)


def test_empty_signal_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        compute_spectrum(np.array([], dtype=complex))


def test_signal_too_short_for_hann_is_rejected():
    with pytest.raises(ValueError, match="too short"):
        compute_spectrum(np.ones(2, dtype=complex), window="hann")
