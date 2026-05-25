#read from .env file
import os
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')

# Load environment variables from .env file
load_dotenv(env_path)

# Access the environment variable
class Config:
    REVIEW_PATH = os.getenv('review_path')
    


