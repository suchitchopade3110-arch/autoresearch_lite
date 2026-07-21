import pytest
import os
import tempfile
import subprocess
from generation.static_check import check_syntax
from generation.patch_generator import validate_and_apply_patch

def test_static_check_valid():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def foo():\n    return 42\n")
        path = f.name

    try:
        success, msg = check_syntax(path)
        assert success is True
        assert msg == ""
    finally:
        os.remove(path)

def test_static_check_invalid():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def foo():\nreturn 42\n") # Indentation error
        path = f.name

    try:
        success, msg = check_syntax(path)
        assert success is False
        assert "IndentationError" in msg or "SyntaxError" in msg
    finally:
        os.remove(path)

def test_validate_and_apply_patch_valid():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=d, check=False)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=d, check=False)

        diff = """--- /dev/null
+++ b/test.py
@@ -0,0 +1,2 @@
+def foo():
+    pass
"""
        # Patch application depends on being in the git repo directory
        cwd = os.getcwd()
        os.chdir(d)
        try:
            assert validate_and_apply_patch(diff) is True
            assert os.path.exists("test.py")
        finally:
            os.chdir(cwd)

def test_validate_and_apply_patch_invalid():
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["git", "init"], cwd=d, check=True)

        diff = """--- a/nonexistent.py
+++ b/nonexistent.py
@@ -1,2 +1,2 @@
-def foo():
+def bar():
"""
        cwd = os.getcwd()
        os.chdir(d)
        try:
            assert validate_and_apply_patch(diff) is False
        finally:
            os.chdir(cwd)
