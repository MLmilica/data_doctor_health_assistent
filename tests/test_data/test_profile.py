"""Tests for data profile generation."""

from data.profile import build_and_save_data_profile, load_data_profile
from data.schema_registry import get_target_columns


def test_build_and_save_data_profile(tmp_path) -> None:
    output_path = tmp_path / "data_profile.json"
    profile, schema, saved_path = build_and_save_data_profile(output_path=output_path)

    assert saved_path == output_path
    assert profile.row_count == 10000
    assert schema.row_count == 10000
    assert len(profile.columns) == len(schema.columns)


def test_profile_contains_targets() -> None:
    profile, _, _ = build_and_save_data_profile()
    profile_column_names = {column.name for column in profile.columns}

    for target_column in get_target_columns():
        assert target_column in profile_column_names


def test_load_saved_data_profile() -> None:
    _, _, saved_path = build_and_save_data_profile()
    loaded = load_data_profile(saved_path)
    assert loaded.row_count == 10000
