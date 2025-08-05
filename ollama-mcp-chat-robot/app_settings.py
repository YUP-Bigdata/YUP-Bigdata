import json
import os

APP_SETTINGS_PATH = "app_settings.json"

DEFAULT_CONFIG = {
    "ai_service": "Ollama",  # Ollama, OpenAI, Claude, Gemini, Qwen
    "llm_model": "llama4:17b-scout-16e-instruct-q8_0",
    "temperature": 0.1,
    "timeout": 600,
    "prompt": "",
}


class AppSettings:
    def __init__(self, config_path=APP_SETTINGS_PATH):
        self.config_path = config_path
        self.config = DEFAULT_CONFIG.copy()
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.config.update(data)

    def save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save()

    def getAll(self):
        self.load()
        return self.config.copy()
