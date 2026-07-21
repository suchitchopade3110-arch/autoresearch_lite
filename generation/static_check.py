import ast
import os
from typing import Tuple

def check_syntax(file_path: str) -> Tuple[bool, str]:
    """
    Checks the syntax of the python file using ast.parse.
    Returns (success, error_message).
    """
    if not os.path.exists(file_path):
        return False, f"File {file_path} does not exist."

    with open(file_path, "r") as f:
        source = f.read()

    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as e:
        error_msg = f"SyntaxError in {file_path}:\nLine {e.lineno}: {e.msg}\n{e.text}"
        return False, error_msg
