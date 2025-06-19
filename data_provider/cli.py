import click
import pandas as pd
from datetime import datetime
import os
from typing import Dict, Any, Optional

#from .config import get_alpha_vantage_api_key
# from .fetchers.alphavantage_fetcher import AlphaVantageFetcher
from .fetchers.yfinance_fetcher import YFinanceFetcher
from .fetchers.demo_fetcher import DemoFetcher
from .data_mapper import DataMapper
from .constants import STRATEGY_COLUMNS, DEFAULT_DATA_FETCHER, \
    DEFAULT_DEMO_NUM_COMPANIES, DEFAULT_DEMO_START_YEAR_OFFSET


def get_fetcher_instance(provider_name: str, cli_kwargs: Dict[str, Any]):
    """
    Instantiates and returns a data fetcher based on the provider name
    and command-line arguments.
    """
    if provider_name == 'demo':
        strategy_name_for_demo = cli_kwargs.get('strategy_template_name')
        if not strategy_name_for_demo:
            # This should be caught by Click's argument handling, but defensive check.
            raise click.UsageError("Strategy template name is required for demo fetcher.")

        return DemoFetcher(
            strategy_template_name=strategy_name_for_demo,
            start_year=cli_kwargs.get('demo_start_year'),
            end_year=cli_kwargs.get('demo_end_year'),
            num_companies=cli_kwargs.get('demo_num_companies'),
            seed=cli_kwargs.get('demo_seed')
        )
    elif provider_name == 'yfinance':
        return YFinanceFetcher()
    # elif provider_name == 'alphavantage':
    #     api_key = cli_kwargs.get('api_key') or get_alpha_vantage_api_key()
    #     if not api_key:
    #         raise click.UsageError(
    #             "API key for Alpha Vantage is required. "
    #             "Provide it via --api-key or set ALPHA_VANTAGE_API_KEY in .env file."
    #         )
    #     return AlphaVantageFetcher(api_key=api_key)
    else:
        raise click.BadParameter(f"Unsupported data provider/fetcher: {provider_name}")


def _generate_default_outputname(strategy_name: str, fetcher_name: str) -> str:
    """
    Generates a default output filename with a counter to avoid overwrites.
    Includes error handling for directory creation and a limit to prevent endless loops.
    """
    base_dir = "data"
    max_attempts = 99  # Safeguard against an endless loop

    try:
        os.makedirs(base_dir, exist_ok=True)  # Ensure the data directory exists
    except OSError as e:
        # Handle potential errors during directory creation (e.g., permission issues)
        raise click.FileError(
            f"Could not create directory '{base_dir}'. Error: {e}",
            hint="Check permissions or if a file with the same name exists."
        )

    counter = 1
    while counter < max_attempts:
        filename = f"{strategy_name}.{fetcher_name}.{counter:02d}.csv"
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

    # If loop finishes, it means no unique filename was found within max_attempts
    raise click.FileError(
        f"Could not generate a unique filename in '{base_dir}' after {max_attempts} attempts. "
        f"Pattern: {strategy_name}.{fetcher_name}.[counter].csv",
        hint="Please check the contents of the 'data' directory or specify an output file manually."
    )



@click.command(name="data")
@click.argument(
    "strategy_template_name",
    metavar="STRATEGY",
    type=click.Choice(list(STRATEGY_COLUMNS.keys()), case_sensitive=False)
)
@click.option(
    "--fetcher", "fetcher_name",
    default=DEFAULT_DATA_FETCHER,
    show_default=True,
    type=click.Choice(['demo', 'yfinance', 'alphavantage'], case_sensitive=False),
    help="The data fetcher to use. 'demo' generates synthetic data."
)
@click.option(
    '--output', '-o',
    type=click.Path(dir_okay=False, writable=True, resolve_path=True),
    default=None,
    help="Path to save the fetched/generated data. If not provided, defaults to './data/[strategy].[fetcher].[counter].csv'."
)
# --- Options for 'demo' fetcher ---
@click.option(
    "--demo-start-year",
    type=int,
    help="[Demo Fetcher] Start year for synthetic data generation."
)
@click.option(
    "--demo-end-year",
    type=int,
    help="[Demo Fetcher] End year for synthetic data generation."
)
@click.option(
    "--demo-num-companies",
    default=DEFAULT_DEMO_NUM_COMPANIES,
    type=click.IntRange(min=1),
    show_default=True,
    help="[Demo Fetcher] Number of companies to generate."
)
@click.option(
    "--demo-seed",
    type=int,
    default=None,
    help="[Demo Fetcher] Random seed for reproducibility."
)
# --- Options for API fetchers (like yfinance, alphavantage) ---
@click.option(
    '--symbols', '-S',
    multiple=True,
    help="[API Fetchers] List of stock symbols (e.g., AAPL MSFT)."
)
@click.option(
    '--api-start-date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="[API Fetchers] Start date for fetching data (YYYY-MM-DD)."
)
@click.option(
    '--api-end-date',
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="[API Fetchers] End date for fetching data (YYYY-MM-DD)."
)
@click.option(
    '--api-key',
    help="[API Fetchers] API key for the provider (if required and not in .env)."
)
@click.pass_context
def data_command(ctx: click.Context, strategy_template_name: str, fetcher_name: str, output: Optional[str],
                 **kwargs: Any):
    """
    Generates or fetches historical market data using the specified fetcher
    and formats it according to the STRATEGY template.
    """
    # Combine explicitly passed args with other kwargs for get_fetcher_instance
    all_cli_params = {
        'strategy_template_name': strategy_template_name,
        'fetcher_name': fetcher_name,
        'output': output,
        **kwargs
    }

    click.echo(f"Initializing data fetcher: {fetcher_name} for strategy template: {strategy_template_name}...")
    try:
        fetcher = get_fetcher_instance(fetcher_name, all_cli_params)
    except (ImportError, click.UsageError, ValueError) as e:
        click.echo(f"Error initializing fetcher: {e}", err=True)
        ctx.exit(1)

    raw_df = pd.DataFrame()

    if fetcher_name == 'demo':
        num_demo_symbols = all_cli_params.get('demo_num_companies', DEFAULT_DEMO_NUM_COMPANIES)
        demo_symbols_list = [f"DEMO{i}" for i in range(num_demo_symbols)]

        s_year = all_cli_params.get('demo_start_year')
        e_year = all_cli_params.get('demo_end_year')

        current_sys_year = datetime.now().year
        effective_end_year = e_year if e_year is not None else current_sys_year - 1
        effective_start_year = s_year if s_year is not None else effective_end_year - DEFAULT_DEMO_START_YEAR_OFFSET

        demo_start_date_str = f"{effective_start_year}-01-01"
        demo_end_date_str = f"{effective_end_year}-12-31"

        click.echo(
            f"Generating demo data for {strategy_template_name} from {effective_start_year} to {effective_end_year}...")
        raw_df = fetcher.fetch_data(
            symbols=demo_symbols_list,
            start_date=demo_start_date_str,
            end_date=demo_end_date_str,
            start_year=all_cli_params.get('demo_start_year'),
            end_year=all_cli_params.get('demo_end_year'),
            num_companies=all_cli_params.get('demo_num_companies'),
            seed=all_cli_params.get('demo_seed')
        )
    elif fetcher_name in ['yfinance', 'alphavantage']:
        api_symbols = all_cli_params.get('symbols')
        api_start_date_dt = all_cli_params.get('api_start_date')
        api_end_date_dt = all_cli_params.get('api_end_date')

        if not api_symbols:
            click.echo("Error: --symbols are required for API fetchers.", err=True)
            ctx.exit(1)
        if not api_start_date_dt or not api_end_date_dt:
            click.echo("Error: --api-start-date and --api-end-date are required for API fetchers.", err=True)
            ctx.exit(1)

        api_start_date_str = api_start_date_dt.strftime("%Y-%m-%d")
        api_end_date_str = api_end_date_dt.strftime("%Y-%m-%d")

        click.echo(
            f"Fetching data for symbols: {', '.join(api_symbols)} from {api_start_date_str} to {api_end_date_str}...")
        raw_df = fetcher.fetch_data(list(api_symbols), api_start_date_str, api_end_date_str)
    else:
        click.echo(f"Fetcher '{fetcher_name}' processing logic not fully implemented in CLI.", err=True)
        ctx.exit(1)

    if raw_df.empty:
        click.echo(f"No data {'generated' if fetcher_name == 'demo' else 'fetched'}. Exiting.", err=True)
        ctx.exit(1)

    click.echo(f"Mapping data to '{strategy_template_name}' format...")
    mapper = DataMapper()
    try:
        final_df = mapper.map_to_strategy_format(raw_df, strategy_template_name, fetcher_name)
    except ValueError as e:
        click.echo(f"Error during data mapping: {e}", err=True)
        ctx.exit(1)

    if final_df.empty:
        click.echo("Data mapping resulted in an empty dataset. Exiting.", err=True)
        ctx.exit(1)

    # Determine output file path.
    if output is None:
        output_path = _generate_default_outputname(strategy_template_name, fetcher_name)
        click.echo(f"No output file specified, using default: {output_path}")
    else:
        output_path = output

    try:
        # Ensure the directory for the output file exists, especially if a custom path is given
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Ensure output_dir is not an empty string (e.g. if output is just a filename)
            os.makedirs(output_dir, exist_ok=True)

        final_df.to_csv(output_path, index=False)
        click.echo(
            f"Successfully {'generated' if fetcher_name == 'demo' else 'fetched'} and saved data to: {output_path}")
        click.echo(f"Output columns: {', '.join(final_df.columns)}")
    except IOError as e:
        click.echo(f"Error saving data to file: {e}", err=True)
        ctx.exit(1)
    except Exception as e:  # Catch any other unexpected errors during file save
        click.echo(f"An unexpected error occurred during file save: {e}", err=True)
        ctx.exit(1)
