import re
from typing import Iterable, Set, Tuple

# Matches the extended git header that names both sides of a rename/copy or
# a brand-new/deleted file: "diff --git a/<old> b/<new>".
_DIFF_GIT_RE = re.compile(r'^diff --git a/(.+) b/(.+)$')
# Traditional unified-diff path lines. git apply accepts either "a/<path>"
# (the default prefix) or a bare path with no prefix at all.
_MINUS_RE = re.compile(r'^--- (?:a/)?(.+?)(?:\t.*)?$')
_PLUS_RE = re.compile(r'^\+\+\+ (?:b/)?(.+?)(?:\t.*)?$')
# "new file mode 120000" / "deleted file mode 120000" is how git apply is
# told to materialize a SYMLINK rather than a regular file - the mode digits
# are a Unix file-type bitmask, not a version number: 100644/100755 are
# regular files, 120000 is a symlink, 160000 is a gitlink (submodule).
_FILE_MODE_RE = re.compile(r'^(?:new|deleted) file mode (\d+)$')
# None of these are ever needed to hand-edit a target file's own content, and
# each is a way to smuggle a change onto a path this guard didn't see named
# in a --- /+++ pair (a rename's source is only in "rename from", not in any
# +++ line) or a change git apply will materialize without any diff hunk at
# all (a permission-bit flip, a binary blob).
_FORBIDDEN_HEADER_RE = re.compile(
    r'^(old mode|new mode|rename from|rename to|copy from|copy to|'
    r'similarity index|dissimilarity index|GIT binary patch)\b'
)
_REGULAR_FILE_MODE_PREFIX = "100"


def _normalize(path: str) -> str:
    return path.strip().replace("\\", "/")


def _is_safe_relative_path(path: str) -> bool:
    if not path or path == "/dev/null":
        return True
    if path.startswith("/"):
        return False
    if len(path) > 1 and path[1] == ":":  # Windows drive letter, e.g. "C:/..."
        return False
    return ".." not in path.split("/")


def validate_diff_scope(diff_text: str, allowed_files: Iterable[str]) -> Tuple[bool, str]:
    """
    Refuses a diff before it ever reaches `git apply` unless every path it
    touches is exactly one of `allowed_files` (config's target.files) and
    the diff only edits plain-file content - no symlinks, no permission-bit
    changes, no renames/copies, no binary patches.

    This is the only thing standing between "a candidate may edit
    candidate_script.py" and "a candidate may turn .eval_out into a symlink
    pointing at .git/hooks, or rewrite eval/pipeline.py to always return
    1.0" - git apply itself enforces neither restriction; it applies
    whatever hunks it's given to whatever paths they name, including
    creating new files and new symlinks outside the paths a caller thinks
    it authorized.

    Returns (ok, reason) - reason is "" when ok is True, otherwise a
    human-readable explanation suitable for a failure_reason/traceback
    field (never raises).
    """
    allowed: Set[str] = {_normalize(f) for f in allowed_files}
    touched: Set[str] = set()

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\r")

        m = _FILE_MODE_RE.match(line)
        if m and not m.group(1).startswith(_REGULAR_FILE_MODE_PREFIX):
            return False, (
                f"diff creates/deletes a non-regular-file mode ({m.group(1)}); "
                "only plain file content changes are permitted"
            )
        if _FORBIDDEN_HEADER_RE.match(line):
            return False, (
                f"diff contains a disallowed header ({line.strip()!r}); only plain "
                "content edits to target.files are permitted"
            )

        m = _DIFF_GIT_RE.match(line)
        if m:
            touched.add(_normalize(m.group(1)))
            touched.add(_normalize(m.group(2)))
            continue

        m = _MINUS_RE.match(line)
        if m:
            touched.add(_normalize(m.group(1)))
            continue

        m = _PLUS_RE.match(line)
        if m:
            touched.add(_normalize(m.group(1)))
            continue

    for path in touched:
        if path == "/dev/null" or path == "":
            continue
        if not _is_safe_relative_path(path):
            return False, f"diff touches a path outside the working tree ({path!r})"
        if path not in allowed:
            return False, (
                f"diff touches {path!r}, which is not in target.files ({sorted(allowed)!r})"
            )

    return True, ""
