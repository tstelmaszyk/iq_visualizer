import numpy as np
import pytest

from iq_visualizer.tone_check import Gap, ToneChecker, format_report


def tone(n_samples, period=16, amplitude=1023.0):
    return amplitude * np.exp(2j * np.pi * np.arange(n_samples) / period)


def test_clean_tone_has_no_gap():
    assert ToneChecker().find_gaps(tone(1000)) == []


def test_quantized_tone_has_no_gap():
    iq = np.round(tone(1000, amplitude=255.0))
    assert ToneChecker().find_gaps(iq) == []


def test_dropped_samples_are_located():
    iq = np.delete(tone(1000), [100, 101, 102])
    assert ToneChecker().find_gaps(iq) == [Gap(start=100, stop=101)]


def test_corrupted_block_is_a_single_gap():
    iq = tone(1000)
    iq[500:520] = 0
    assert ToneChecker().find_gaps(iq) == [Gap(start=500, stop=521)]


def test_constant_data_longer_than_the_tone_is_a_gap():
    iq = np.concatenate((tone(400), np.full(600, 5000 + 5000j)))
    assert ToneChecker().find_gaps(iq) == [Gap(start=400, stop=1000)]


def test_signal_without_rotation_is_rejected():
    with pytest.raises(ValueError):
        ToneChecker().find_gaps(np.zeros(100, dtype=complex))


def test_amplitude_change_is_a_gap():
    iq = tone(1000)
    iq[300] *= 0.5
    assert ToneChecker().find_gaps(iq) == [Gap(start=300, stop=301)]


def test_several_gaps_are_reported_in_order():
    iq = np.delete(tone(1000), [100, 600])
    assert ToneChecker().find_gaps(iq) == [Gap(100, 101), Gap(599, 600)]


def test_gap_at_the_end_is_closed():
    iq = tone(100)
    iq[-5:] = 0
    assert ToneChecker().find_gaps(iq) == [Gap(start=95, stop=100)]


@pytest.mark.parametrize("n_samples", [0, 1])
def test_too_short_signal_is_rejected(n_samples):
    with pytest.raises(ValueError):
        ToneChecker().find_gaps(tone(n_samples))


def test_report_without_gap():
    assert format_report([]) == "Tone check: no gap detected."


def test_report_lists_gaps_in_samples():
    report = format_report([Gap(32, 34), Gap(100, 101)])
    assert report.splitlines() == [
        "Tone check: 2 gap(s) detected.",
        "  samples 32-33 (2 samples)",
        "  samples 100-100 (1 samples)",
    ]


def test_report_adds_time_when_sample_rate_is_known():
    report = format_report([Gap(1000, 1010)], sample_rate=1e6)
    assert report.splitlines()[1] == (
        "  samples 1000-1009 (10 samples), t = 0.001 s")
