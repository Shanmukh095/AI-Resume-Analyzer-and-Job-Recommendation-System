import re

def clean_resume(text: str) -> str:

    text = text.replace("－", "-")

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()
