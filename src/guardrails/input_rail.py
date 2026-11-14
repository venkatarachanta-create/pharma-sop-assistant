import re


BLOCKED_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"jailbreak",
    r"bypass\s+(rules|safety|guardrails)",
    r"\b(prescribe|prescription|diagnose|treat\s+my|dosage\s+for\s+me)\b",
]


def check_input_guardrail(query: str) -> tuple[bool, str]:
    cleaned = query.strip()
    if not cleaned:
        return False, "Please enter a valid regulatory or SOP question."

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            return False, (
                "⛔ **Input Guardrail Triggered:** I cannot provide personal medical advice, "
                "prescriptions, or override compliance instructions. Ask strictly about "
                "uploaded CDSCO/GMP/GDP documents."
            )

    return True, cleaned
