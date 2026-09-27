import os
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

from observability.logging_config import get_logger

_module_logger = get_logger(__name__)


class SandboxExecutor:
    """
    Executes a script inside a Docker sandbox.

    SECURITY NOTE:
    Runs as a non-root user with no network access, a read-only root
    filesystem, and dropped capabilities, on top of the wall-clock
    timeout and CPU/memory limits. This raises the bar against a
    candidate script trying to exfiltrate data, persist state, or exceed
    its resource limits, but it is still a standard container, not a
    hardened micro-VM - it does NOT protect against a deliberate kernel
    exploit or container escape. Do not run untrusted malware here.
    """
    def __init__(self, config: Dict[str, Any], dataset_dir: Optional[str] = None):
        self.timeout = config.get('timeout_seconds', 10)
        self.cpu_limit = config.get('cpu_limit', '1.0')
        self.memory_limit = config.get('memory_limit', '512m')
        # pids_limit caps how many processes/threads a candidate can fork -
        # without it, a fork bomb can exhaust the host's PID table even
        # though CPU/memory are capped, since neither limit bounds process
        # *count*. ulimit_nofile caps open file descriptors per-process for
        # the same reason (a descriptor-exhaustion loop isn't CPU/memory-
        # bound either). tmpfs_size_mb bounds the writable /tmp mount so it
        # can't be filled up to exhaust host RAM (tmpfs is backed by RAM).
        self.pids_limit = config.get('pids_limit', 128)
        self.ulimit_nofile = config.get('ulimit_nofile', 1024)
        self.tmpfs_size_mb = config.get('tmpfs_size_mb', 64)
        # /app/out is a real host bind mount (rw), not a tmpfs - Docker has
        # no --tmpfs-style size flag for a bind mount, so nothing on the
        # `docker run` command line can cap it the way tmpfs_size_mb caps
        # /tmp. A candidate that writes large junk files into it (beyond
        # predictions.jsonl itself, which eval/pipeline.py already caps at
        # MAX_PREDICTIONS_FILE_BYTES before reading it) could otherwise fill
        # host disk with no limit at all. Enforced host-side, after the
        # container exits, by summing out_dir's real file sizes.
        self.out_dir_max_mb = config.get('out_dir_max_mb', 256)
        # Opt-in only, never enabled by default - GPU passthrough (via the
        # NVIDIA Container Toolkit) is an isolation trade-off the operator
        # must choose explicitly, not something this harness should decide
        # on their behalf. A value like "all" or "device=0" is passed
        # straight through to `docker run --gpus`; leave unset (None) to
        # keep candidates with no GPU access at all, same as today.
        self.gpus = config.get('gpus')
        if self.gpus:
            _module_logger.warning(
                f"sandbox.gpus={self.gpus!r} is set - candidates get GPU access via --gpus. "
                "This requires the NVIDIA Container Toolkit on the host and reduces the "
                "sandbox's isolation guarantees (a GPU driver is a much larger, less "
                "audited attack surface than the CPU-only path). Enable only if you trust "
                "the candidates being generated."
            )
        # train.jsonl/test.jsonl mounted read-only into every sandbox run if
        # set, so callers (the sequential loop and the concurrent
        # evolutionary scheduler alike) don't each need to know about
        # dataset wiring individually. The held-out labels file living
        # alongside them is NEVER mounted here - only the host-side eval
        # pipeline reads it, so a candidate can never read its own answer
        # key off disk. See eval/dataset.py and eval/pipeline.py.
        self.dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else None
        self._build_image()

    def _build_image(self):
        # Resolved relative to this file, not the caller's CWD - a
        # pip-installed `autoresearch` is invoked from whatever directory
        # the operator happens to be in, which has no reason to contain a
        # sandbox/Dockerfile of its own. This directory (Dockerfile +
        # requirements-sandbox.txt) is bundled as package data - see
        # pyproject.toml's [tool.setuptools.package-data].
        sandbox_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(
            ["docker", "build", "-t", "ml-sandbox", "-f", os.path.join(sandbox_dir, "Dockerfile"), sandbox_dir],
            check=True,
            capture_output=True
        )

    def run_candidate(self, script_path: str, env_vars: Optional[Dict[str, str]] = None,
                       out_dir: Optional[str] = None, extra_files: Optional[List[str]] = None,
                       train_path_override: Optional[str] = None,
                       test_path_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Runs the given script inside the docker sandbox. extra_files are
        additional worktree paths (for multi-file candidates - see
        target.files in config_schema.py) each bind-mounted read-only into
        /app/<basename>, alongside the primary candidate_script.py mount -
        scoped explicitly per-file the same way that mount already is,
        never as a mount of the whole worktree directory (which would also
        expose .git and anything else sitting in there).

        train_path_override, if given, is mounted at /app/data/train.jsonl
        INSTEAD OF dataset_dir/train.jsonl - the caller's way of actually
        enforcing progressive-scaling subsets (see eval/dataset.py:
        write_subset). Without it, every stage mounts the SAME full
        training file regardless of SUBSET_PERCENTAGE, which is then only
        an environment variable a candidate's own code may simply ignore.
        test.jsonl is never subsetted - the test set stays whole at every
        stage so scores remain comparable across stages (see
        eval/dataset.py:subset_indices).

        test_path_override, if given, is mounted at /app/data/test.jsonl
        INSTEAD OF dataset_dir/test.jsonl - used for exactly one purpose:
        the sealed-holdout evaluation run (see eval/dataset.py's
        generate_split docstring and orchestrator/run.py's
        _score_holdout). Every progressive-scaling stage during the loop
        itself sees the SAME test.jsonl (the "selection" set) every time,
        by design - repeatedly gating merge decisions on one fixed set lets
        the baseline ratchet upward on that set's own sampling noise rather
        than genuine improvement. The holdout set is scored exactly once,
        after a candidate has already cleared every stage and the baseline
        gate on the selection set, and that holdout score is never itself
        used to gate anything - it is reported alongside the selection
        score for a human reviewer (or an auditor) to see the difference.
        """
        start_time = time.time()
        container_name = f"sandbox-{uuid.uuid4().hex[:8]}"

        cmd = self._build_docker_cmd(
            script_path, container_name, env_vars=env_vars, out_dir=out_dir,
            extra_files=extra_files, train_path_override=train_path_override,
            test_path_override=test_path_override,
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            execution_time = time.time() - start_time

            if out_dir is not None:
                oversized, out_dir_size = self._out_dir_exceeds_cap(out_dir)
                if oversized:
                    return {
                        "exit_code": -1,
                        "stdout": result.stdout,
                        "stderr": (
                            f"Candidate wrote {out_dir_size} bytes to /app/out, exceeding the "
                            f"{self.out_dir_max_mb}MB cap - execution result discarded."
                        ),
                        "execution_time": execution_time,
                        "timeout": False,
                    }

            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time": execution_time,
                "timeout": False
            }
        except subprocess.TimeoutExpired as e:
            execution_time = time.time() - start_time
            self._force_stop_container(container_name)
            return {
                "exit_code": -1,
                "stdout": e.stdout.decode() if e.stdout else "",
                "stderr": e.stderr.decode() if e.stderr else f"Execution timed out after {self.timeout} seconds.",
                "execution_time": execution_time,
                "timeout": True
            }

    def _force_stop_container(self, container_name: str) -> None:
        """
        Second-audit-round finding: `subprocess.run`'s own timeout only
        kills the LOCAL `docker run` client process - the docker DAEMON
        manages the container independently and can keep it running after
        the client disconnects, so a bare client-side kill is not
        sufficient. `docker stop` (bounded by --time so a wedged container
        can't make this call itself hang) sends SIGTERM then SIGKILL after
        its grace period; `docker kill` is a fallback for the case `stop`
        itself fails or the daemon doesn't act on it. Every call here is
        itself bounded by `timeout=` (subprocess.run's own arg) so neither
        can hang indefinitely if the docker daemon itself is unresponsive.
        Best-effort and silent on failure - the caller already has a real
        timeout result to return regardless of whether this cleanup
        succeeds; any container this fails to reap is caught by
        `cleanup_orphan_containers` at the next startup instead.
        """
        for cmd in (
            ["docker", "stop", "--time", "5", container_name],
            ["docker", "kill", container_name],
        ):
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=15)
                if result.returncode == 0:
                    return
            except subprocess.TimeoutExpired:
                continue
        _module_logger.warning(
            f"Could not confirm container {container_name!r} was stopped after its execution timed "
            "out - it may still be running. It will be reaped at the next startup's crash-recovery "
            "cleanup (see cleanup_orphan_containers) if it's still present then."
        )


    def _out_dir_exceeds_cap(self, out_dir: str) -> "tuple[bool, int]":
        """
        Sums the real on-disk size of every regular file under out_dir
        (the sandbox's rw /app/out bind mount) and compares it against
        out_dir_max_mb. Symlinks are not followed (os.walk's default
        followlinks=False, and os.path.getsize on a symlink reports the
        link's own tiny size, never the target's) - a candidate trying to
        inflate the reported size via a symlink to a large host file gains
        nothing here; the real defense against a symlink at this path is
        the file-safety check in eval/pipeline.py:load_predictions and the
        symlink guard in _build_docker_cmd above. Returns (exceeded, total_bytes).
        """
        cap_bytes = self.out_dir_max_mb * 1024 * 1024
        total = 0
        for root, _dirs, files in os.walk(out_dir):
            for name in files:
                path = os.path.join(root, name)
                if os.path.islink(path):
                    continue
                try:
                    total += os.path.getsize(path)
                except OSError:
                    continue
                if total > cap_bytes:
                    return True, total
        return False, total

    def _build_docker_cmd(self, script_path: str, container_name: str,
                           env_vars: Optional[Dict[str, str]] = None,
                           out_dir: Optional[str] = None, extra_files: Optional[List[str]] = None,
                           train_path_override: Optional[str] = None,
                           test_path_override: Optional[str] = None) -> List[str]:
        """
        Builds the full `docker run` argument list for one candidate
        execution - every bind mount `run_candidate` passes to Docker is
        decided here, and only here. Kept separate from run_candidate (which
        actually invokes Docker) so tests can assert on the real mount
        arguments a run would use - e.g. that the held-out labels file is
        never among them - without needing a Docker daemon at all; see
        tests/test_reward_hacking.py.

        Never mounts the held-out truth.json labels file (see eval/dataset.py
        and eval/pipeline.py) - only train.jsonl/test.jsonl from dataset_dir,
        or train_path_override in place of train.jsonl, ever appear in a -v
        argument built here.
        """
        # Defense in depth against a candidate-controlled path (e.g. inside
        # a worktree) being a symlink: os.chmod and Docker's bind-mount
        # source resolution both follow symlinks, so chmod-ing or mounting
        # one silently operates on whatever it points at instead of the
        # file the caller thinks it's granting sandbox access to. The real
        # gate against a candidate ever creating such a symlink in the
        # first place is vcs/diff_guard.py; this is a second, independent
        # check at the point the mount is actually constructed, so a gap or
        # future bypass in the diff guard can't silently regress this too.
        for candidate_path in [script_path, *(extra_files or [])]:
            if os.path.islink(candidate_path):
                raise ValueError(f"Refusing to mount a symlink into the sandbox: {candidate_path}")
        if train_path_override and os.path.islink(train_path_override):
            raise ValueError(f"Refusing to mount a symlink into the sandbox: {train_path_override}")
        if test_path_override and os.path.islink(test_path_override):
            raise ValueError(f"Refusing to mount a symlink into the sandbox: {test_path_override}")

        # A bind mount carries the HOST file's real permission bits into the
        # container - the sandbox's UID (1000) is never the host process's
        # own UID, so a file created with a restrictive mode (e.g. 0600,
        # which tempfile.NamedTemporaryFile uses by default) is unreadable
        # to it. This is silently masked on Docker Desktop for Windows/macOS
        # (whose VM-backed bind mounts don't enforce host permission bits
        # the same way) but fails immediately on native Linux Docker - so
        # never assume the caller already got this right.
        os.chmod(script_path, 0o644)
        for extra_path in (extra_files or []):
            os.chmod(extra_path, 0o644)

        cmd = [
            "docker", "run", "--rm",
            f"--name={container_name}",
            f"--cpus={self.cpu_limit}",
            f"--memory={self.memory_limit}",
            f"--pids-limit={self.pids_limit}",
            "--ulimit", f"nofile={self.ulimit_nofile}",
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:size={self.tmpfs_size_mb}m",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "-v", f"{script_path}:/app/candidate_script.py:ro",
        ]

        for extra_path in (extra_files or []):
            cmd += ["-v", f"{extra_path}:/app/{os.path.basename(extra_path)}:ro"]

        if self.gpus:
            cmd += ["--gpus", self.gpus]

        run_env = dict(env_vars or {})
        if self.dataset_dir or train_path_override:
            train_path = train_path_override or os.path.join(self.dataset_dir, "train.jsonl")
            cmd += ["-v", f"{os.path.abspath(train_path)}:/app/data/train.jsonl:ro"]
            run_env.setdefault("TRAIN_PATH", "/app/data/train.jsonl")
        if self.dataset_dir or test_path_override:
            test_path = test_path_override or os.path.join(self.dataset_dir, "test.jsonl")
            cmd += ["-v", f"{os.path.abspath(test_path)}:/app/data/test.jsonl:ro"]
            run_env.setdefault("TEST_PATH", "/app/data/test.jsonl")

        if out_dir:
            if os.path.islink(out_dir):
                raise ValueError(f"Refusing to mount a symlink into the sandbox: {out_dir}")
            os.makedirs(out_dir, exist_ok=True)
            # os.makedirs uses the process umask, typically leaving a
            # directory writable only by its owner (0755) - the sandbox's
            # UID needs to create predictions.jsonl inside it, so it must
            # be writable by everyone, not just whichever UID happened to
            # create it host-side. Same host-vs-container UID mismatch as
            # the script mount above.
            os.chmod(out_dir, 0o777)
            # A read-write bind mount coexists fine with --read-only on the
            # root filesystem - only this path is writable.
            cmd += ["-v", f"{os.path.abspath(out_dir)}:/app/out:rw"]

        for key, value in run_env.items():
            cmd += ["-e", f"{key}={value}"]

        cmd.append("ml-sandbox")
        return cmd


def cleanup_orphan_containers() -> int:
    """
    Removes every container named "sandbox-*" (SandboxExecutor.run_candidate's
    own naming convention) - crash recovery for a previous orchestrator
    process that was killed before SandboxExecutor._force_stop_container
    ever got a chance to run, leaving a candidate's container running (or
    merely present but stopped) indefinitely. Meant to be called once, at
    startup, before any candidate of THIS run has been scheduled - same
    timing contract as vcs/git_controller.py:cleanup_orphans.

    A module-level function, not a SandboxExecutor method, so the
    standalone `cleanup` subcommand (and startup cleanup in general) can
    reap orphaned containers without constructing a full SandboxExecutor -
    whose __init__ builds the Docker image, an expensive and unnecessary
    step for what should be a fast, cheap cleanup pass (and one that would
    hard-fail outright wherever Docker isn't installed at all).

    CONCURRENT RUNS: containers are host-global, not scoped to a git repo
    the way worktrees are - a second orchestrator process running on the
    SAME HOST (even against a different repo_path) would have its own
    in-flight candidate containers removed by this too. This is the same
    class of risk vcs/git_controller.py:cleanup_orphans already accepts
    and documents for concurrent runs against the same repo_path, just
    with a wider blast radius (host-wide, not repo-scoped) - do not run
    two orchestrator processes on the same host concurrently.

    Best-effort: any docker error here (including Docker not being
    installed/running at all) is logged and swallowed, never raised - this
    must never block a run from starting. Returns the number of containers
    removed.
    """
    try:
        listing = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=^sandbox-"],
            capture_output=True, text=True, timeout=15, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        _module_logger.warning(f"Could not list orphan sandbox containers for cleanup: {e}")
        return 0

    container_ids = [line for line in listing.stdout.splitlines() if line.strip()]
    removed = 0
    for container_id in container_ids:
        try:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=15, check=True)
            removed += 1
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            _module_logger.warning(f"Could not remove orphan sandbox container {container_id!r}")
    return removed
