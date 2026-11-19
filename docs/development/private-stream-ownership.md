# Private stream ownership for large checked C entries

New `guarded-inplace-large` modules require a runtime with
`PRIVATE_STREAM_OWNERSHIP_VERSION == 1`, `create_stream()`, and asynchronous
`retire_completed_stream(stream)`. The check runs before stream creation,
including frames with no active tiles. The ordinary five-buffer C entry and
`make_plan` / `execute_plan` success interfaces are unchanged. Other compiler
profiles continue to use their existing stream behavior.

The prepared plan creates each exact stream object before dispatch and retains
it with the runtime, source, destination, and scratch buffers. Its generated tile
and checked/private entry receive that object through an internal keyword-only
`_adora_stream` argument. The mapper's one `runtime.create_stream()` assignment
is replaced in the selected private entry only. Helpers and emitter output
remain unchanged. The generator rejects a different stream-creation shape
rather than guessing which stream it owns.

The owned binding also requires the emitter's known final synchronization and
writeback tail. It removes the tail's `except NameError: pass` compatibility
guard and calls `apply_writeback_tasks(writeback_tasks)` directly. A missing
helper or any error inside final writeback therefore propagates as an execution
failure before public publication or retirement; the plan keeps unknown-output
storage poisoned and rooted. Other profiles retain their existing guard.

A tile returns only after its generated synchronization and writeback. The
prepared plan copies `old_dst[:end-begin]` to the public destination, advances its
cursor and `completed_ranges`, then calls
`await runtime.retire_completed_stream(stream)`. Returned slots are copied once,
including inactive slots; the host does not reevaluate device stores. The final
partial tile copies only its valid range.

## Cleanup failures and retry

A retirement exception raises `PlanCleanupError` with these attributes:

| Attribute | Meaning |
| --- | --- |
| `phase` | `retire_completed_stream` |
| `block` | The published block's `(begin, end)` range |
| `completed_ranges` | The already published prefix snapshot |
| `cursor` | The count of published active blocks |
| `cause` / `__cause__` | The original retirement exception |
| `plan` | The retained ownership and retry handle |

`await error.retry_cleanup()` or `await error.plan.retry_cleanup()` retries only
retirement of the exact stream. It does not dispatch, transfer, publish, or
advance the cursor again. Repeated calls after success do nothing. A retry
through a saved error remains bound to its original cleanup identity:
after that cleanup succeeds, the old error is a no-op even when the same plan
has another frame awaiting cleanup. The explicit `plan.retry_cleanup()` method
targets the plan's current pending cleanup. Concurrent
retry on the same plan raises `ValueError` while the first retry owns cleanup;
the runtime also makes exact-object retirement idempotent, including a retry
after retirement took effect but its caller observed an error.

While cleanup is pending, `plan.cleanup_pending` is true and
`plan.pending_stream` exposes the exact object. `plan.close()` and frame reuse
reject until cleanup succeeds. A process-rooted pending set retains the plan,
runtime and borrowed arrays if callers drop their local handles. A successful
retry removes that root and clears the frame borrow. The once entry preserves
the error and its plan handle; it does not close a plan with pending cleanup.
Its close operation cannot replace an earlier execution failure: a close error
is attached to that original exception as `close_error`.

Retirement cancellation keeps its cancellation/control-flow class, enriched
with `phase`, `block`, `completed_ranges` and `plan`; cleanup remains retryable.
Preparation and public copyback failures remain `PlanExecutionError`, never
`PlanCleanupError`. If generated work returned before public copyback failed,
the stream remains safely available to `plan.retry_cleanup()` without trying
the failed publication again.

## Unknown completion

A generated tile exception or cancellation does not establish completion. The
plan retains its exact stream, runtime, source, destination and scratch storage,
marks itself poisoned, and preserves only previously published ranges. It does
not retire the stream. Close, reuse, and cleanup retry reject unknown completion.
An unscheduled coroutine remains in the existing in-flight process root even if
its scheduler never injects an exception. A late child completion does not
implicitly publish data or clear the borrow. The runtime version marker is a
capability check, never completion evidence.

## Runtime and cache compatibility

The runtime retirement contract validates both owning runtime and exact object
identity. It accepts only successfully completed accepted work, releases each
stream's resources once, and keeps no normal history table. Explicit user
streams remain reusable until their owner closes them. `synchronize_all()`
snapshots registered streams and their accepted work before its first suspension;
concurrent retirement preserves those completion credentials and later streams
do not extend that wait. These guarantees belong to the paired shared runtime.

Older generated modules remain supported by the new runtime. New modules reject
older runtimes before creating a stream. Descriptor/support versions 2 and 3
remain accepted; ownership is an explicit binding option, separate from value
semantics and descriptor compatibility. The compiler always regenerates the
checked/private binding and embeds current prepared support after loading raw
mapper cache entries. Module identity includes production Python source hashes,
so a baseline-populated mapper cache cannot preserve an older ownership binding.

The Python regressions in `test/adoracc/private_stream_ownership/` exercise the
host protocol using synthetic transport; they do not establish hardware
execution correctness. End-to-end validation additionally requires the actual
shared runtime, paired hardware configuration, real done/D2H, and public arrays.

Run the shared runtime software regressions with
`python -B runtime/python/test_stream_ownership.py` in the simulation
environment. For real device execution and host readback use the named
short-tail and ownership cases in the [full RTL instructions](../../README.md#optional-full-rtl-simulation).
