import os
from dotenv import dotenv_values
from pathlib import Path

def setup_config_dictionary(env_path="config.env"):
    '''
    keys ending in _DIR assumed to be directories, we'll create fallback directories for them if they don't exist.
    '''
    all_keys = { key: os.getenv(key) for key in dotenv_values(env_path) }
    for key, value in all_keys.items():
        if key.endswith("_DIR"):
            # If the directory does not exist, create a local directory with the same name
            if not os.path.exists(value):
                dir_name = Path(value).name
                fallback_path = Path.cwd() / dir_name
                fallback_path.mkdir(exist_ok=True)
                all_keys[key] = str(fallback_path)

    return all_keys