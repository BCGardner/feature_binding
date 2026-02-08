import pytest
from typing import Optional, Dict, Any

from hsnn.utils.io import formatted_name


class TestFormattedName:
    """Tests for the formatted_name function.
    """

    @pytest.mark.parametrize(
        "base_name, ext, expected",
        [
            ("file", None, "file"),
            ("file", "txt", "file.txt"),
            ("file", ".log", "file.log"),
            ("report", "csv", "report.csv"),
            ("", "ext", ".ext"), # Test empty base name
        ]
    )
    def test_basic_name_and_extension(self, base_name: str, ext: Optional[str], expected: str):
        """Test basic functionality with base name and optional extension."""
        assert formatted_name(base_name, ext=ext) == expected

    @pytest.mark.parametrize(
        "base_name, kwargs, expected",
        [
            ("data", {"enabled": True}, "data_enabled"),
            ("data", {"enabled": False}, "data"),
            ("config", {"debug": True, "verbose": True}, "config_debug_verbose"),
            ("config", {"debug": True, "test": False, "verbose": True}, "config_debug_verbose"),
            ("run", {"active": True, "inactive": False}, "run_active"),
        ]
    )
    def test_boolean_kwargs(self, base_name: str, kwargs: Dict[str, Any], expected: str):
        """Test handling of boolean keyword arguments."""
        assert formatted_name(base_name, **kwargs) == expected

    @pytest.mark.parametrize(
        "base_name, kwargs, expected",
        [
            ("result", {"count": 10}, "result_count_10"),
            ("result", {"value": 5.5}, "result_value_5_5"),
            ("result", {"count": 0}, "result"),
            ("result", {"value": 0.0}, "result"),
            ("result", {"count": -5}, "result"),
            ("result", {"value": -1.2}, "result"),
            ("stats", {"epoch": 5, "batch": 100}, "stats_epoch_5_batch_100"),
            ("stats", {"epoch": 0, "batch": 100, "lr": 0.1}, "stats_batch_100_lr_0_1"),
        ]
    )
    def test_numeric_kwargs(self, base_name: str, kwargs: Dict[str, Any], expected: str):
        """Test handling of numeric keyword arguments."""
        assert formatted_name(base_name, **kwargs) == expected

    @pytest.mark.parametrize(
        "base_name, kwargs, expected",
        [
            ("model", {"type": "resnet"}, "model_type_resnet"),
            ("output", {"user": "test", "mode": "eval"}, "output_user_test_mode_eval"),
            ("log", {"level": "info"}, "log_level_info"),
            ("file", {"empty_str": ""}, "file_empty_str_"), # Test empty string value
        ]
    )
    def test_other_kwargs(self, base_name: str, kwargs: Dict[str, Any], expected: str):
        """Test handling of other (e.g., string) keyword arguments."""
        assert formatted_name(base_name, **kwargs) == expected

    @pytest.mark.parametrize(
        "base_name, ext, kwargs, expected",
        [
            (
                "experiment",
                "log",
                {"debug": True, "version": 3, "lr": 0.001, "name": "final"},
                "experiment_debug_version_3_lr_0_001_name_final.log"
            ),
            (
                "run",
                ".dat",
                {"fast": True, "slow": False, "iter": 50, "skip": 0, "rate": 1.5, "label": "test"},
                "run_fast_iter_50_rate_1_5_label_test.dat"
            ),
            (
                "output",
                "csv",
                {"processed": True, "threshold": 0.95, "invalid": -10, "user": "admin"},
                "output_processed_threshold_0_95_user_admin.csv"
            ),
            ( # Test no kwargs
                "simple",
                "txt",
                {},
                "simple.txt"
            ),
            ( # Test no extension
                "noext",
                None,
                {"flag": True, "value": 1},
                "noext_flag_value_1"
            ),
            ( # Test empty base name with args and ext
                "",
                "conf",
                {"a": True, "b": 10, "c": "setup"},
                "_a_b_10_c_setup.conf"
            )
        ]
    )
    def test_mixed_args_and_extension(self, base_name: str, ext: Optional[str], kwargs: Dict[str, Any], expected: str):
        """Test handling of mixed keyword argument types and extension."""
        assert formatted_name(base_name, ext=ext, **kwargs) == expected
