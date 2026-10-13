import os
import requests
import polars as pl
from dotenv import load_dotenv

load_dotenv()


def get_data() -> pl.DataFrame:
    """Fetch sales data from the API endpoint into a Polars DataFrame."""
    url = os.getenv("API_URL")
    user = os.getenv("API_USERNAME")
    password = os.getenv("API_PASSWORD")

    response = requests.get(url, auth=(user, password), timeout=30)
    response.raise_for_status()
    return pl.DataFrame(response.json())


class MainIngestion:
    def get_data(self) -> pl.DataFrame:
        return get_data()
