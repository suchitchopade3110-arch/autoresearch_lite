def generate_candidate() -> str:
    """
    Generates a candidate script.

    TODO: Phase 2 - Implement LLM-based code generation.
    Currently returns a dummy/no-op script.
    """
    return "print('Hello from candidate script', flush=True)\nprint('MOCK_SCORE: 0.99', flush=True)\n"
