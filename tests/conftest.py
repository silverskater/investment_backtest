import pytest
from backtest.example_data import generate_example_data  # Ensure this import path is correct


@pytest.fixture(scope="session")  # Session scope is efficient as data generation logic is consistent
def generic_data_file_factory(tmp_path_factory):
    """
    A generic factory fixture to create temporary sample data files for testing.

    This factory can generate data for different strategies and allows for
    custom data modification.
    """
    created_files_and_dirs = []

    def _create_data_file(
            strategy_name: str,
            start_year: int = 2020,
            end_year: int = 2022,
            num_companies: int = 10,
            seed: int = 42,
            custom_data_modifier=None,  # Optional function to modify data before saving
            filename_prefix: str = "sample_data"
    ):
        # Create a unique directory for each call to ensure isolation and easier cleanup
        # numbered=True helps if the same parameters are called multiple times in complex scenarios
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
        created_files_and_dirs.append(data_dir)  # Also track the directory for cleanup
        return str(file_path)

    yield _create_data_file

    # Clean up all created files and then directories
    for path_obj in reversed(created_files_and_dirs):  # Remove files before dirs
        if path_obj.is_file():
            try:
                path_obj.unlink()
            except OSError:
                # Log or handle error if necessary, e.g., file locked
                pass
        elif path_obj.is_dir():
            try:
                # Only remove if empty, or use shutil.rmtree if forceful removal is needed (with caution)
                if not any(path_obj.iterdir()):  # Check if directory is empty
                    path_obj.rmdir()
            except OSError:
                # Log or handle error, e.g., directory not empty or permission issues
                pass
