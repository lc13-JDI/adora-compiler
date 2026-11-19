# Stream ownership and completion

`DeviceRuntime.PRIVATE_STREAM_OWNERSHIP_VERSION == 1` advertises this contract.
The module exports the same constant. New generated private callers check the
capability before creating a stream or submitting work.

An owner keeps the exact object returned by `runtime.create_stream()`. After
`await stream.synchronize()` and successful generated host writeback, the caller
may publish its valid output range, then call
`await runtime.retire_completed_stream(stream)`. Retirement never waits for
unknown work: it requires successful completion of all accepted commands and
rejects the stream's original work error. Cross-runtime objects and same-ID
impostors raise `ValueError`. New submissions during or after retirement raise
`RuntimeError`. Concurrent retirement serializes on a per-stream lock; repeated
retirement of that same object is harmless. Idempotence is held on the stream
object, with no runtime tombstone/history registry.

The existing misspelled `await runtime.destory_stream(stream)` remains an owner
close operation. It validates exact identity, waits for accepted work, and then
retires. A user-owned explicit stream remains reusable until its owner closes
it. `is_device_busy()` keeps its existing meaning: at least one stream remains
registered, including an idle stream owned by an external caller.

`synchronize()` waits for work accepted when its coroutine begins execution.
`synchronize_all()` snapshots every current member and every member's accepted
work before its first suspension. Each snapshot holds independent completion
evidence, so concurrent retirement is safe. Later submissions, including later
work on an existing member, are excluded. Internal FIFO completion markers flush
reordered transfers and confirm any running execution. Empty/retired streams do
not start workers. `_accepted`, `_completed`, `_error`, `_retired`, `_retiring`
and transient `_fences` are diagnostic state, not public APIs.

Work, transfer and allocator errors wake waiters with the original exception.
Bounded transfer admission rechecks failure after its suspended queue put:
newly inserted entries are failed and removed synchronously, waking every other
blocked producer to reject with the same original error. A failed transfer
queue cannot restart a worker or launch another DMA; rejected entries do not
remain queued. This includes producers already blocked when the error arose.
Execution completion requires a fresh matching done response; absent done or
stale status raises `TimeoutError` after 10000 polling intervals and leaves
`handler.running` true. Failure does not retire a stream, release a running
resource, mark completion, or restart a failed worker. The owner must preserve
the stream and its borrowed buffers when completion is unknown; retirement is
not a recovery API for failed work. Cleanup failures leave the exact registered
object available for retry. A returned valid allocation is invalidated before
notification/logging so a cleanup retry cannot return it twice.

Run the software regression with the production cocotb Python environment and
Icarus Verilog available on `PATH`:

```sh
python server/test_stream_ownership.py
```

The self-launcher creates and removes a temporary empty Verilog top. Tests run
real cocotb 1.9 queues/events/tasks and the real runtime methods; hardware
allocation/status interfaces use minimal doubles. Exit status is derived from
the XML results because cocotb 1.9's make target can return zero for test failure.
Set `RUNTIME_UNDER_TEST=/absolute/path/to/test_runif.py` to test a frozen older
source; the same assertions are expected to fail there. These software tests do
not establish RTL or generated-entry correctness; the complete RTL gates and
private-call publication/borrowing checks must be run separately.
