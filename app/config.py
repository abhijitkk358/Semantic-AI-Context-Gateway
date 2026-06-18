import os
from dotenv import load_dotenv

# Load key-value pairs from the hidden .env file into os.environ
load_dotenv()

class Settings:
    """
    Decoupled application environment configuration manager.
    Prevents hardcoding target server routing definitions into core logic.
    """
    # Phase 1 maps to a stable, free test endpoint to simulate a downstream LLM service
    MOCK_LLM_URL: str = os.getenv(
        "MOCK_LLM_URL", 
        "https://jsonplaceholder.typicode.com/posts/1"
    )

# CRITICAL LINE: This must match what main.py is importing!
settings = Settings()
