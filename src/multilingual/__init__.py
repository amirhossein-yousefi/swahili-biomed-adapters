"""multilingual: MAD-X-style adapter stacking for Swahili biomedical NLP."""
__version__ = "0.1.0"

# Load .env (HF_TOKEN, WANDB_API_KEY, etc.) before anything else imports HF/wandb
from .utils.env import load_dotenv as _load_dotenv  # noqa: E402

_load_dotenv()
