FROM python:3.9-slim

WORKDIR /app

# Note: This sandbox provides basic resource isolation and standard container boundaries.
# It does NOT provide hardened security against deliberate container escapes or zero-days.
# It is designed to constrain CPU/memory and enforce timeouts, not to run untrusted malware safely.

COPY . /app

CMD ["python", "candidate_script.py"]
