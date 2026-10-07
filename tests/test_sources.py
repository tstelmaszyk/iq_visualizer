import numpy as np
import pytest

from iq_visualizer.config import CsvConfig
from iq_visualizer.sources import CsvIQSource, IQSource


def read_csv_text(tmp_path, text, **csv_options):
    path = tmp_path / "iq.csv"
    path.write_text(text)
    return CsvIQSource(CsvConfig(path=path, **csv_options)).read()


def test_csv_source_is_an_iq_source(tmp_path):
    assert isinstance(CsvIQSource(CsvConfig(path=tmp_path)), IQSource)


def test_reads_comma_separated_values(tmp_path):
    iq = read_csv_text(tmp_path, "1,2\n3,-4\n")
    np.testing.assert_array_equal(iq, [1 + 2j, 3 - 4j])
    assert np.iscomplexobj(iq)


def test_reads_semicolon_and_decimal_comma(tmp_path):
    iq = read_csv_text(tmp_path, "1,5;2\n3;-4,25\n",
                       separator=";", decimal=",")
    np.testing.assert_array_equal(iq, [1.5 + 2j, 3 - 4.25j])


def test_reads_whitespace_separated_values(tmp_path):
    iq = read_csv_text(tmp_path, "1 2\n3    4\n", separator=r"\s+")
    np.testing.assert_array_equal(iq, [1 + 2j, 3 + 4j])


def test_reads_file_with_header(tmp_path):
    iq = read_csv_text(tmp_path, "I,Q\n1,2\n", has_header=True)
    np.testing.assert_array_equal(iq, [1 + 2j])


def test_reads_non_contiguous_columns_in_requested_order(tmp_path):
    iq = read_csv_text(tmp_path, "0,10,99,20\n1,11,99,21\n",
                       i_column=3, q_column=1)
    np.testing.assert_array_equal(iq, [20 + 10j, 21 + 11j])


def test_undeclared_header_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="not numeric"):
        read_csv_text(tmp_path, "I,Q\n1,2\n")


def test_wrong_separator_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="separator"):
        read_csv_text(tmp_path, "1;2\n3;4\n")


def test_empty_file_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        read_csv_text(tmp_path, "")


def test_header_only_file_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="No samples"):
        read_csv_text(tmp_path, "I,Q\n", has_header=True)


def test_missing_value_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        read_csv_text(tmp_path, "1,2\n3,\n")


def test_missing_file_raises_file_not_found(tmp_path):
    source = CsvIQSource(CsvConfig(path=tmp_path / "absent.csv"))
    with pytest.raises(FileNotFoundError):
        source.read()


def test_infinite_value_gives_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="infinite"):
        read_csv_text(tmp_path, "1,2\n3,inf\n")
