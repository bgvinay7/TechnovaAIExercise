import json
from pathlib import Path

DATASET_PATH = Path(__file__).with_name("training_data.jsonl")


def validate_dataset() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {DATASET_PATH}")

    count = 0
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"Blank line at line {line_number}")

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Line {line_number}: messages must be a non-empty list"
                )

            roles = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"Line {line_number}: each message must be an object"
                    )
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise ValueError(
                        f"Line {line_number}: unsupported role {role!r}"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise ValueError(
                        f"Line {line_number}: message content must be non-empty text"
                    )
                roles.append(role)

            if "user" not in roles or "assistant" not in roles:
                raise ValueError(
                    f"Line {line_number}: each example needs user and assistant messages"
                )
            count += 1

    if count == 0:
        raise ValueError("No training examples found.")
    print(f"Dataset validation successful: {count} examples.")
    print("Note: provider-specific requirements may still apply.")
    return count


if __name__ == "__main__":
    validate_dataset()
