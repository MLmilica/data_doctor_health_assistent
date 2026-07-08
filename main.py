"""CLI entry point for local utilities."""

from data.profile import build_and_save_data_profile


def main() -> None:
    profile, schema, path = build_and_save_data_profile()
    print(f"Rows: {profile.row_count}")
    print(f"Columns: {len(schema.columns)}")
    print(f"Saved data profile to: {path}")


if __name__ == "__main__":
    main()
