import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def get_text_provider(provider: Optional[str] = None) -> str:
    return (provider or os.getenv("TEXT_MODEL_PROVIDER") or "aigcbest").strip().lower()


def get_image_provider(provider: Optional[str] = None) -> str:
    return (provider or os.getenv("IMAGE_MODEL_PROVIDER") or "legacy").strip().lower()


def get_text_model_config(provider: Optional[str] = None) -> Dict[str, str]:
    provider_name = get_text_provider(provider)

    if provider_name == "yunwu":
        return {
            "provider": "yunwu",
            "model": os.getenv("YUNWU_TEXT_MODEL", "gemini-3-flash-preview-all"),
            "api_key": os.getenv("YUNWU_API_KEY") or os.getenv("GEMINI_API_KEY", ""),
            "base_url": os.getenv("YUNWU_BASE_URL", "https://yunwu.ai/v1"),
        }

    if provider_name == "deepseek":
        return {
            "provider": "deepseek",
            "model": os.getenv("DEEPSEEK_TEXT_MODEL", "deepseek-chat"),
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        }

    return {
        "provider": "aigcbest",
        "model": os.getenv("GEMINI_TEXT_MODEL", "gemini-3-flash-preview-all"),
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "base_url": os.getenv("GEMINI_BASE_URL", "https://api2.aigcbest.top/v1"),
    }


def get_text_api_keys(provider: Optional[str] = None) -> List[str]:
    config = get_text_model_config(provider)
    env_map = {
        "yunwu": "YUNWU_API_KEYS",
        "deepseek": "DEEPSEEK_API_KEYS",
        "aigcbest": "GEMINI_API_KEYS",
    }
    raw = os.getenv(env_map.get(config["provider"], "GEMINI_API_KEYS"), "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if keys:
        return keys
    if config["api_key"]:
        return [config["api_key"]]
    
    # 如果当前提供商的API key为空，回退到aigcbest
    if config["provider"] != "aigcbest":
        print(f"⚠️ 警告: {config['provider']} 提供商的API key未配置，回退到 aigcbest")
        return get_text_api_keys("aigcbest")
    
    return []


def get_image_api_keys(provider: Optional[str] = None) -> List[str]:
    provider_name = get_image_provider(provider)
    if provider_name == "yunwu_openai":
        raw = (
            os.getenv("YUNWU_OPENAI_IMAGE_API_KEYS")
            or os.getenv("YUNWU_OPENAI_API_KEYS")
            or os.getenv("YUNWU_IMAGE_API_KEYS")
            or os.getenv("YUNWU_API_KEYS")
            or ""
        )
    elif provider_name == "yunwu":
        raw = os.getenv("YUNWU_IMAGE_API_KEYS", os.getenv("YUNWU_API_KEYS", ""))
    else:
        raw = os.getenv("SORA_API_KEYS", "")
    
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    
    # 如果当前提供商的API key为空，回退到legacy
    if not keys and provider_name in {"yunwu", "yunwu_openai"}:
        print(f"⚠️ 警告: yunwu 图像提供商的API key未配置，回退到 legacy")
        return get_image_api_keys("legacy")
    
    return keys
