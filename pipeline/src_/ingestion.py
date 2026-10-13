import requests
import polars as pl
from dotenv import load_dotenv
import os 
from requests.auth import HTTPBasicAuth

load_dotenv()


import os
import requests
import polars as pl
from requests.auth import HTTPBasicAuth


class MainIngestion:
    def __init__(self):
        self.api_url = os.getenv("API_URL")
        self.username = os.getenv("API_USERNAME")
        self.password = os.getenv("API_PASSWORD")

    def send_request(self):
        response = requests.get(
            self.api_url,
            auth=HTTPBasicAuth(self.username, self.password),
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def get_data(self):
        data = self.send_request()
        return pl.DataFrame(data)

    
