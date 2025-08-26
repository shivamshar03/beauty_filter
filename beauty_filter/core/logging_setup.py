
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: str) -> logging.Logger:
    """Setup comprehensive logging system"""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        log_filename = f"beauty_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(log_dir, log_filename)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info("Logging system initialized")
        return logger

    except Exception as e:
        print(f"[ERROR] Failed to setup logging: {e}")
        # Fallback to basic console logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
