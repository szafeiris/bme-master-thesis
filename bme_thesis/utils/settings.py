from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class BmeThesisSettings(BaseSettings):
    # Path definitions
    base_path: str = ''
    images_path: str = ''
    masks_path: str = ''
    
    # Logging settings
    log_level: str = "DEBUG"
    log_interval: str = "d"
    log_interval_count: int = 1
    log_backup_count: int = 10
    log_path: str = "logs"
    log_filename: str = "bme-master-thesis.log"

    # Pyradiomics Params.yml files
    pyradiomics_params_file: str = "./data/resources/Params.yaml"

    # Telegram settings
    telegram_token: str = ""
    telegram_chat_id: Optional[int] = None
    telegram_send: bool = False
    
    model_config = SettingsConfigDict(extra='ignore')

bmeThesisSettings = BmeThesisSettings(_env_file=f'environments\.env.{os.getenv("TARGET")}')
    
__all__ = [BmeThesisSettings, bmeThesisSettings]
