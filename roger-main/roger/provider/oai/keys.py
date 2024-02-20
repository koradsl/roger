from typing import Dict


def get_key_params(api_key: str, org_key: str) -> Dict[str, str]:
    if api_key and org_key:
        return {"api_key": api_key, "organization": org_key}

    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    org_key = os.getenv("OPENAI_ORG_KEY", "")

    assert api_key != "", "Invalid OpenAI API Key"
    assert org_key != "", "Invalid OpenAI ORG Key"

    return {"api_key": api_key, "organization": org_key}
