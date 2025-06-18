"""Utility functions for loading and preparing market data."""
import json
import os

import click
import pandas as pd
from pandas.errors import EmptyDataError

from .stress_tests import apply_stress_test


def prepare_market_data(
        data_file_path: str,
        stress_test: str,
        start_year: int,
        end_year: int,
        include_delisted: bool
) -> pd.DataFrame:
    """Loads and preprocesses market data for backtesting.

    This function loads data from the specified file, applies any selected
    stress test scenarios, and handles options related to delisted companies.

    Args:
        data_file_path: Path to the CSV or JSON file containing market data.
        stress_test: Name of the stress test scenario to apply (e.g., 'none',
                     '2008crisis').
        start_year: The start year of the backtest period.
        end_year: The end year of the backtest period.
        include_delisted: Boolean flag to indicate whether to include data for
                          delisted companies.

    Returns:
        A pandas DataFrame containing the prepared market data.
    """
    click.echo(f"Loading market data from {data_file_path}...")
    full_market_data = _load_market_data(data_file_path)

    if stress_test != 'none':
        click.echo(f"Applying stress test scenario: {stress_test} for period {start_year}-{end_year}")
        full_market_data = apply_stress_test(
            full_market_data, stress_test, start_year, end_year
        )

    if include_delisted:
        if 'is_delisted' in full_market_data.columns:
            click.echo("Including delisted companies in analysis.")
        else:
            click.echo(
                "Warning: Delisted company data ('is_delisted' column) "
                "not available in the provided market data.",
                err=True
            )
    return full_market_data


def _load_market_data(file_path: str) -> pd.DataFrame:
    """Loads market data from a CSV or JSON file.

    Args:
        file_path: The path to the data file.

    Returns:
        A pandas DataFrame containing the market data.

    Raises:
        FileNotFoundError: If the specified `file_path` does not exist.
        ValueError: If the file format is unsupported (not CSV or JSON),
                    or if there's an error processing the file content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        try:
            return pd.read_csv(file_path)
        except EmptyDataError:
            click.echo(f"Warning: CSV file {file_path} is empty. Returning empty DataFrame.", err=True)
            return pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error processing CSV file {file_path}: {e}") from e
    elif file_extension == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            if isinstance(data_list, list) and not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error processing JSON file {file_path}: {e}") from e
    else:
        raise ValueError(
            f"Unsupported file format: '{file_extension}'. Please use CSV or JSON."
        )