import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from backtest.utils.quality_utils import (
    map_sp_quality_to_numeric,
    SP_QUALITY_RANK_MAPPING,
    MIN_SP_QUALITY_NUMERIC
)


class TestQualityUtils:
    """Test suite for quality_utils.py."""

    def test_sp_quality_rank_mapping_content(self):
        """Tests the content and integrity of SP_QUALITY_RANK_MAPPING.

        Ensures the mapping is a dictionary, contains expected key-value pairs,
        and only includes the explicitly defined quality ranks.
        """
        assert isinstance(SP_QUALITY_RANK_MAPPING, dict)

        # Verify specific important key-value pairs are correct
        assert SP_QUALITY_RANK_MAPPING['A+'] == 6
        assert SP_QUALITY_RANK_MAPPING['B+'] == 3
        assert SP_QUALITY_RANK_MAPPING['NR'] == 0
        assert SP_QUALITY_RANK_MAPPING['C'] == 0
        assert SP_QUALITY_RANK_MAPPING['D'] == -1

        # Define the complete set of all expected keys
        expected_keys = {'A+', 'A', 'A-', 'B+', 'B', 'B-', 'C', 'D', 'NR'}

        # Assert that the keys in the mapping are exactly the expected keys
        # This ensures no undefined ranks are present and all defined ranks exist.
        assert set(SP_QUALITY_RANK_MAPPING.keys()) == expected_keys, (
            "SP_QUALITY_RANK_MAPPING keys do not match the expected set. "
            f"Expected: {sorted(list(expected_keys))}, "
            f"Got: {sorted(list(SP_QUALITY_RANK_MAPPING.keys()))}"
        )

    def test_min_sp_quality_numeric_value(self):
        """Tests the value of MIN_SP_QUALITY_NUMERIC."""
        assert isinstance(MIN_SP_QUALITY_NUMERIC, int)
        assert MIN_SP_QUALITY_NUMERIC == SP_QUALITY_RANK_MAPPING['B+']
        assert MIN_SP_QUALITY_NUMERIC == 3

    @pytest.mark.parametrize(
        "input_series, expected_series",
        [
            # Test case 1: Basic valid mappings
            (
                pd.Series(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C', 'D', 'NR']),
                pd.Series([6, 5, 4, 3, 2, 1, 0, -1, 0], dtype=int)
            ),
            # Test case 2: Mixed valid, invalid, and NaN values
            (
                pd.Series(['A+', 'InvalidRank', None, 'B-', 'NR', float('nan')]),
                pd.Series([6, 0, 0, 1, 0, 0], dtype=int)
            ),
            # Test case 3: All invalid or NaN
            (
                pd.Series(['X', 'Y', None, float('nan')]),
                pd.Series([0, 0, 0, 0], dtype=int)
            ),
            # Test case 4: Empty series
            (
                pd.Series([], dtype=str),
                pd.Series([], dtype=int)
            ),
            # Test case 5: Series with only NaN
            (
                pd.Series([None, float('nan')], dtype=object), # dtype=object for None
                pd.Series([0, 0], dtype=int)
            ),
            # Test case 6: Series with numeric-like strings (should be unmapped)
            (
                pd.Series(['1', '2.0']),
                pd.Series([0, 0], dtype=int)
            )
        ]
    )
    def test_map_sp_quality_to_numeric(self, input_series, expected_series):
        """Tests map_sp_quality_to_numeric with various inputs."""
        result_series = map_sp_quality_to_numeric(input_series)
        assert_series_equal(result_series, expected_series, check_dtype=True)

    def test_map_sp_quality_to_numeric_dtype(self):
        """Ensures the output series has integer dtype."""
        input_s = pd.Series(['A+', 'B'])
        result_s = map_sp_quality_to_numeric(input_s)
        assert result_s.dtype == int, f"Expected dtype int, got {result_s.dtype}"

        input_s_empty = pd.Series([], dtype=str)
        result_s_empty = map_sp_quality_to_numeric(input_s_empty)
        assert result_s_empty.dtype == int, f"Expected dtype int for empty series, got {result_s_empty.dtype}"