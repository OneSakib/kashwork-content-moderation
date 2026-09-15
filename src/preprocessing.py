import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


if __name__ == "__main__":
    print("==", normalize_text("  Premium LAPTOP   with 16GB RAM  "))
