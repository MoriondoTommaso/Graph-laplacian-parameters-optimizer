import multiprocessing as mp
import numpy as np


def _build_worker(graph_params: dict, items: np.ndarray, result_queue: mp.Queue) -> None:
    # This function runs ONLY inside the spawned subprocess.
    # The Rust allocator owns all memory created here.
    # When this subprocess exits, the OS reclaims everything atomically —
    # no conflict with the parent process's Python GC.
    try:
        # Import arrowspace INSIDE the worker, not at module level.
        # This ensures the Rust extension is loaded fresh in the child process,
        # never inherited from the parent via fork.
        from arrowspace import ArrowSpaceBuilder

        # items.copy() creates a new numpy buffer owned exclusively by this
        # subprocess. Without .copy(), the numpy array's memory could be
        # shared with the parent, causing a double-free when both sides
        # try to release it.
        aspace, _gl = ArrowSpaceBuilder().build(graph_params, items.copy())

        # lambdas_sorted() returns the non-zero eigenvalues of the Laplacian
        # in ascending order: [λ₁, λ₂, λ₃, ...].
        # λ₁ is the Fiedler value. λ₂ - λ₁ is the spectral gap.
        lambdas = aspace.lambdas_sorted()

        # Send the result back to the parent process through the Queue.
        # Queue uses pickle serialisation — only plain Python objects cross
        # the process boundary. No Rust pointers, no numpy arrays.
        result_queue.put({"lambdas": list(lambdas)})

    except Exception as e:
        # Any failure (degenerate graph, Rust panic, bad params) is caught here
        # and sent back as an error string. The subprocess exits cleanly.
        # The parent process receives {"error": ...} instead of crashing.
        result_queue.put({"error": str(e)})


def run_isolated_build(graph_params: dict, items: np.ndarray) -> list[float] | None:
    # This is the only function the rest of the codebase calls.
    # It never touches ArrowSpaceBuilder directly — it only manages
    # the subprocess lifecycle and reads from the Queue.

    # "spawn" creates a brand-new Python interpreter for the child.
    # This is the critical safety choice: the child has no knowledge of
    # any Rust-allocated objects in the parent's memory space.
    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    # Pass graph_params and items to the worker via constructor arguments.
    # These are serialised through pickle when the process spawns.
    p = ctx.Process(target=_build_worker, args=(graph_params, items, q))
    p.start()

    # Wait up to 60 seconds for the build to complete.
    # If the subprocess hangs (e.g., Rust deadlock), we don't block forever.
    p.join(timeout=60)

    # A non-zero exit code means the subprocess crashed (e.g., Rust panic
    # that was not caught by the try/except). Return None so Optuna
    # can prune this trial without the main process dying.
    if p.exitcode != 0:
        return None

    # Read the result from the Queue without blocking.
    # get_nowait() raises queue.Empty if nothing was put — treated as failure.
    try:
        result = q.get_nowait()
    except Exception:
        return None

    # If the worker sent back an error dict, return None gracefully.
    if "error" in result:
        return None

    # Happy path: return the list of eigenvalues to the caller.
    return [item[0] for item in result["lambdas"]]
