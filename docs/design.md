# Design

## Scope

The runtime owns the boundary from a complete sensor observation to a checked,
time-indexed action chunk. It does not own camera drivers, robot middleware,
network transport, model conversion, or low-level tensor kernels.

Keeping these concerns outside the core prevents CUDA, ROS, and serialization
dependencies from leaking into a robot control process that may not need them.

## Contracts

`Observation` owns its data so an asynchronous request cannot outlive camera or
middleware buffers. A future zero-copy adapter can use a backend-specific
reference-counted tensor in `extra_inputs`, but borrowed raw pointers are not
part of the default API.

`ModelSpec` is the resolved contract of a loaded backend. The engine validates
state and action shapes against it. `ActionChunk` carries the originating
request, control step, timestamp, execution period, and representation.

`Status` and `Result<T>` keep expected data and timing failures out of exception
paths. Constructors still throw for programmer errors such as a null backend.

## Scheduling

VLA inference is usually slower than a control tick. An unbounded FIFO queue
makes every later result stale, so the default queue capacity is one and
`keep_latest` cancels the pending observation when a newer one arrives. The
currently executing backend call is not forcefully interrupted; a backend can
cooperate through `CancellationToken` and `deadline`.

Backend access is serialized because common inference contexts and GPU streams
are not re-entrant. Parallelism should be expressed as multiple engines with
separate backend contexts or by a batching backend.

## Action execution

User processors run in insertion order, followed by the built-in observation
validator. A delta policy should install
`DeltaToAbsolute` before `ActionSafetyFilter`; limits and rate checks then
operate in the actuator coordinate system.

`ActionBuffer` indexes actions by control step. A newer request replaces an
older prediction at overlapping future steps. Once a step is requested, all
earlier actions are discarded and cannot be reintroduced by an out-of-order
chunk.

The buffer does not implement policy-specific temporal ensembling or RTC
inpainting. Those algorithms need confidence weights or flow-policy prefix
conditioning and should be separate processors/backends rather than implicit
averaging in the generic buffer.

## Thread safety

`Scheduler`, `ActionBuffer`, and metrics are thread-safe. Adding processors is
configuration-time work and must finish before concurrent calls begin. An
`Engine` serializes inference, processor state, and reset calls.

## Failure policy

Malformed observations, NaN/Inf values, invalid model output shapes, missed
deadlines, cancellations, and queue overflow are explicit statuses. A caller
must decide its robot-specific fallback: hold position, use the remaining
buffered trajectory, invoke a classical controller, or enter a safe stop.
