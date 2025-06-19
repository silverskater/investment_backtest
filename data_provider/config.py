import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

def get_alpha_vantage_api_key() -> str | None:
    return os.getenv("ALPHA_VANTAGE_API_KEY")

# Add functions for other API keys as needed