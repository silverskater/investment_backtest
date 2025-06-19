"""Pytest configuration and shared fixtures for the test suite."""
import pytest
from data_provider.fetchers.demo_fetcher import _generate_demo_data_logic as generate_example_data


@pytest.fixture(scope="session")
def generic_data_file_factory(tmp_path_factory):
    """Creates temporary sample data files for testing.

    This session-scoped factory fixture generates data for different strategies
    and allows for custom data modification before saving to a temporary CSV file.
    It also handles cleanup of created files and directories.

    Yields:
        A factory function `_create_data_file` that can be called by tests.
    """
    created_files_and_dirs = []

    def _create_data_file(
            strategy_name: str,
            start_year: int = 2020,
            end_year: int = 2022,
            num_companies: int = 10,
            seed: int = 42,
            custom_data_modifier=None,
            filename_prefix: str = "sample_data"
    ):
        """Generates and saves a temporary data file.

        Args:
            strategy_name: Name of the strategy for data generation.
            start_year: Start year for the data.
            end_year: End year for the data.
            num_companies: Number of companies to generate.
            seed: Random seed for reproducibility.
            custom_data_modifier: Optional function to modify the DataFrame
                                  before saving.
            filename_prefix: Prefix for the generated filename.

        Returns:
            The string path to the created temporary data file.
        """
        data_dir = tmp_path_factory.mktemp(f"data_{strategy_name}_{seed}_", numbered=True)
        file_path = data_dir / f"{filename_prefix}_{strategy_name}_{seed}.csv"

        df = generate_example_data(
            strategy_name=strategy_name,
            start_year=start_year,
            end_year=end_year,
            num_companies=num_companies,
            seed=seed
        )

        if custom_data_modifier:
            df = custom_data_modifier(df)

        df.to_csv(file_path, index=False)
        created_files_and_dirs.append(file_path)
        created_files_and_dirs.append(data_dir)
        return str(file_path)

    yield _create_data_file

    # Cleanup: remove files before directories.
    for path_obj in reversed(created_files_and_dirs):
        if path_obj.is_file():
            try:
                path_obj.unlink()
            except OSError:
                # Log or handle error if necessary (e.g., file locked).
                pass
        elif path_obj.is_dir():
            try:
                if not any(path_obj.iterdir()):  # Check if directory is empty.
                    path_obj.rmdir()
            except OSError:
                # Log or handle error (e.g., directory not empty or permission issues).
                pass