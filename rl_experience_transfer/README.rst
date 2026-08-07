RL Experience Transfer
======================

Move reinforcement-learning experience from rollout workers to training workers without coupling
either side to one transport or framework.

``rl-experience-transfer`` provides a versioned canonical schema, safe serialization, explicit
acknowledgement and retry semantics, durable duplicate suppression, consumer compatibility
contracts, and adapters for NeMo RL, PRIME-RL, slime, and MILES. The Python import name is
``rlxfer``.

The library transfers experience data--tokens, rewards, log probabilities, advantages,
trajectories, and metadata. It does not transfer model weights or implement a rollout scheduler,
replay buffer, or trainer.

Why use it?
-----------

* Keep rollout and training code independent of the data plane.
* Preserve framework-native fields without using pickle or executable deserialization.
* Reject incompatible algorithms, tokenizers, models, policy versions, or schemas before training.
* Use the same producer/consumer API with memory, filesystem, or NIXL transports.
* Make delivery outcomes explicit with ack, retrying nack, terminal rejection, and receipts.
* Persist consumed idempotency keys and content-free dead letters across consumer restarts.
* Add HMAC authentication and W3C trace context without changing the canonical schema.

Quick start
-----------

Install from a source checkout:

.. code-block:: console

   python -m pip install .

For development, ``uv`` creates the locked environment used by CI:

.. code-block:: console

   uv sync --locked --dev

This complete in-process example publishes, receives, validates, and acknowledges one batch:

.. code-block:: python

   import numpy as np

   from rlxfer import (
       ExperienceBatch,
       ExperienceConsumer,
       ExperienceMetadata,
       ExperienceProducer,
       ReceiptState,
       TensorPayload,
       Trajectory,
   )
   from rlxfer.transports import InMemoryTransport

   transport = InMemoryTransport()
   producer = ExperienceProducer(transport)
   consumer = ExperienceConsumer(transport)

   batch = ExperienceBatch(
       metadata=ExperienceMetadata(
           producer_id="rollout-0",
           producer_framework="custom",
           producer_framework_version="1.0",
       ),
       trajectories=(
           Trajectory(
               tokens=TensorPayload(np.asarray([101, 102, 103], dtype=np.int64)),
               rewards={"task": 1.0},
           ),
       ),
   )

   receipt = producer.publish(batch, idempotency_key="rollout-0:sample-42")
   delivery = consumer.receive(timeout=1.0)
   if delivery is None:
       raise TimeoutError("experience was not delivered")

   # Run the training step before acknowledging durable success.
   assert delivery.batch.trajectories[0].rewards["task"] == 1.0
   delivery.ack()
   assert receipt.wait(timeout=1.0).state is ReceiptState.ACKED
   transport.close()

For separate processes on one host, construct ``FileSystemTransport`` instances with the same
queue directory. See ``examples/filesystem_process.py`` for a runnable spawned-process example.

Framework-native data
---------------------

Adapters convert native rollout records into ``ExperienceBatch`` and reconstruct native training
inputs at the consumer. Framework imports are optional and happen only during native conversion.

.. code-block:: python

   from rlxfer import ExperienceConsumer, ExperienceProducer
   from rlxfer.adapters import create_adapter
   from rlxfer.transports import FileSystemTransport

   queue = FileSystemTransport("/var/lib/rlxfer/queue")
   adapter = create_adapter("slime")
   producer = ExperienceProducer(queue, adapter=adapter)
   consumer = ExperienceConsumer(queue, adapter=adapter)

   receipt = producer.publish(native_rollout_output)
   delivery = consumer.receive(timeout=10.0)
   if delivery is None:
       raise TimeoutError("experience was not delivered")
   native_training_input = delivery.to_framework()
   # trainer.step(native_training_input)
   delivery.ack()

Install the framework itself in its project-supported environment, then install this package into
that environment. Framework-named extras are intentionally dependency-free placeholders because
the audited frameworks do not share one portable PyTorch/CUDA dependency solution.

The adapters fail closed outside their audited native API ranges:

.. list-table:: Audited adapter compatibility
   :header-rows: 1

   * - Framework
     - Native versions
     - Verified revision
   * - NeMo RL
     - 0.5.x through 0.7.x
     - ``daf46ff`` / ``81aa43d``
   * - PRIME-RL
     - 0.5.x
     - ``2873bf2``
   * - slime
     - 0.3.x
     - ``a6272da``
   * - MILES
     - 0.2.x
     - ``319716c``

Native conversion matrix
~~~~~~~~~~~~~~~~~~~~~~~~

Self-roundtrips are supported for all four adapters. slime and MILES can also exchange native
records when the consumer supplies explicit, matching semantic requirements. Other cross-framework
pairings are rejected because their preserved native fields do not provide a lossless target
representation.

.. list-table:: Producer to consumer conversion
   :header-rows: 1

   * - Producer
     - NeMo RL
     - PRIME-RL
     - slime
     - MILES
   * - NeMo RL
     - Supported
     - Rejected
     - Rejected
     - Rejected
   * - PRIME-RL
     - Rejected
     - Supported
     - Rejected
     - Rejected
   * - slime
     - Rejected
     - Rejected
     - Supported
     - Supported with contract
   * - MILES
     - Rejected
     - Rejected
     - Supported with contract
     - Supported

Run one real native integration or the complete 4-by-4 contract matrix after installing the pinned
framework sources:

.. code-block:: console

   python examples/framework_roundtrip.py slime
   python examples/framework_roundtrip.py slime --consumer-framework miles
   python examples/framework_matrix.py

These commands use real native record classes, filesystem serialization, native reconstruction, a
finite PyTorch backward pass, a verified optimizer update, and terminal settlement. They are adapter
integration tests, not full framework trainer runs.

Pinned framework pipeline components can also be wired at test time without patching the upstream
repositories:

.. code-block:: console

   python examples/framework_pipeline.py all --format markdown

This CPU-safe harness verifies the source path of every claimed upstream component, publishes with
``ExperienceProducer``, consumes with ``ExperienceConsumer``, runs the framework-owned loss, updates
a tiny policy, and acknowledges only after the update:

.. list-table:: Framework-owned CI component paths
   :header-rows: 1

   * - Framework
     - Rollout-side boundary
     - Training-side boundary
     - CPU stand-in
   * - NeMo RL
     - ``RolloutManager.generate_and_push``
     - ``ClippedPGLossFn``
     - Deterministic generation and environment
   * - PRIME-RL
     - ``TrainingBatchSender.send``
     - ``compute_loss``
     - Synthetic native batch
   * - slime
     - ``call_rollout_fn``
     - ``compute_policy_loss``
     - Test-time rollout plugin
   * - MILES
     - ``call_rollout_fn``
     - ``compute_policy_loss``
     - Test-time rollout plugin

The stand-ins replace external model servers, environments, or GPU-distributed engines; the listed
orchestration, plugin, native-type, and loss code is imported from the pinned upstream checkout.
This is real component integration, but it is not presented as a complete distributed trainer run.

Choose a transport
------------------

.. list-table:: Built-in transports
   :header-rows: 1

   * - Transport
     - Use it for
     - Persistence
     - Notes
   * - ``InMemoryTransport``
     - Tests and threads in one process
     - No
     - Bounded queue, failure injection, at-least-once until process exit
   * - ``FileSystemTransport``
     - Processes on one host
     - Yes
     - Atomic private files, fsync, leases, stale-inflight recovery
   * - ``NixlTransport``
     - Registered CPU/CUDA buffers
     - Control plane only
     - Optional Linux dependency; UCX data plane with explicit buffer lifetime
   * - ``FallbackTransport``
     - Ordered transport failover
     - Child-dependent
     - Advertises the conservative intersection of child capabilities

Transport capabilities are checked before publication with ``TransferPlan``. Fallback retries only
errors known to occur before acceptance by default; broadening its exception list can duplicate a
delivery if an error was raised after acceptance.

For the optional NIXL data plane:

.. code-block:: console

   python -m pip install ".[nixl]"
   python examples/benchmark_nixl.py --device cpu

Reliable delivery
-----------------

Delivery is at least once. Exactly-once processing is not claimed. Use a stable idempotency key for
the logical training item and acknowledge only after the training-side effect is durable.

.. code-block:: python

   delivery = consumer.receive(timeout=10.0)
   if delivery is not None:
       try:
           train_durably(delivery.batch)
       except TemporaryTrainerError as error:
           delivery.nack(str(error), retry=True)
       except IncompatibleBatchError as error:
           delivery.reject(str(error))
       else:
           delivery.ack()

``SqliteDeliveryState`` persists consumed keys and dead letters across restarts:

.. code-block:: python

   from rlxfer import ExperienceConsumer, SqliteDeliveryState

   consumer = ExperienceConsumer(
       transport,
       state_store=SqliteDeliveryState("/var/lib/rlxfer/delivery.sqlite"),
   )

Dead-letter records contain IDs, attempt counts, timestamps, and reasons--never prompts, responses,
or tensor payloads. ``examples/reliable_transfer.py`` composes durable state, retry, authentication,
consumer contracts, policy staleness, tracing, and filesystem delivery in one runnable example.

Consumer contracts
------------------

Use the same contract at publication and consumption for preflight validation and defense in depth:

.. code-block:: python

   from rlxfer import CompatibilityRequirements, ConsumerContract, PolicyVersion

   contract = ConsumerContract(
       CompatibilityRequirements(
           consumer_framework="trainer",
           consumer_framework_version="1.0",
           algorithm="grpo",
           tokenizer_id="tokenizer-revision",
           model_id="model-revision",
           policy_version=PolicyVersion(120, policy_id="actor", model_id="model-revision"),
           max_policy_lag=2,
       ),
       required_fields=frozenset({"trajectories.tokens"}),
   )

Contracts can require algorithm, tokenizer, model, reward definition, sequence format, padding,
chat template, truncation, policy identity/staleness, reference log probabilities, and dotted field
paths. Schema changes are applied only through explicit functions registered in an instance-scoped
``SchemaMigrationRegistry``.

Authentication and tracing
--------------------------

``AuthenticatedExperienceSerializer`` signs the canonical metadata and every external tensor
buffer with HMAC-SHA256. Key IDs allow rotation; key material is supplied out of band.

.. code-block:: python

   import os

   from rlxfer import AuthenticatedExperienceSerializer

   key = bytes.fromhex(os.environ["RLXFER_HMAC_KEY_HEX"])
   serializer = AuthenticatedExperienceSerializer(
       {"2026-08": key},
       signing_key_id="2026-08",
   )
   producer = ExperienceProducer(transport, serializer=serializer)
   consumer = ExperienceConsumer(transport, serializer=serializer)

Authentication detects tampering and the wrong key; it does not encrypt payloads. Protect queue
storage and network links separately when experience contains sensitive data. Authentication reads
every buffer and can trade away a direct-buffer fast path.

Use ``with_trace_context`` and ``trace_context_from`` to propagate validated W3C ``traceparent`` and
``tracestate`` values through adapters and transports.

Production checklist
--------------------

* Assign a stable idempotency key to every logical training item.
* Use ``SqliteDeliveryState`` or an application implementation of ``DeliveryStateStore``.
* Declare a ``ConsumerContract`` instead of accepting unspecified training semantics.
* Set bounded publish/receive/receipt timeouts and a finite retry budget.
* Treat permanent incompatibility as ``reject``; reserve ``nack(retry=True)`` for transient errors.
* Authenticate transfers that cross a trust boundary and encrypt sensitive storage and links.
* Monitor health, receipt states, retries, dead letters, queue depth, and end-to-end latency.
* Test the exact framework revision and accelerator stack used in production.
* Keep producer buffers alive and immutable until terminal settlement when using NIXL.
* Plan for duplicate delivery; acknowledgement alone is not distributed exactly-once execution.

Examples
--------

.. list-table:: Runnable examples
   :header-rows: 1

   * - Command
     - Demonstrates
   * - ``python examples/basic.py``
     - Minimal canonical publish, receive, and acknowledgement
   * - ``python examples/filesystem_process.py``
     - Persistent transfer between spawned processes
   * - ``python examples/reliable_transfer.py``
     - Authentication, trace context, retry, contract, and durable state
   * - ``python examples/cross_framework.py``
     - Actionable semantic incompatibility reporting
   * - ``python examples/framework_roundtrip.py slime``
     - One real framework-native adapter roundtrip and optimizer update
   * - ``python examples/framework_matrix.py``
     - All 16 declared framework pairings, including expected safe rejections
   * - ``python examples/framework_pipeline.py all --format markdown``
     - Pinned framework rollout boundaries, transfer lifecycle, losses, and optimizer updates
   * - ``python examples/benchmark_nixl.py --device cpu``
     - Byte-exact NIXL transfer with machine-readable timings

Develop and test
----------------

The dependency-light suite supports Python 3.10 and 3.12:

.. code-block:: console

   uv sync --locked --dev
   uv run --locked pytest -m "not nixl and not requires_gpu and not requires_xpu"
   uv run --locked ruff check src tests examples
   uv run --locked ruff format --check src tests examples
   uv run --locked mypy
   uv build

Tests are marked ``unit``, ``integration``, ``e2e``, ``multi_process``, ``nixl``, ``requires_gpu``,
``requires_xpu``, and by framework. CI runs the core suite on Python 3.10 and 3.12, builds and installs
the wheel in an isolated environment, and runs the full native 4-by-4 adapter contract matrix
against pinned framework source revisions on CPU. The same job runs four framework-owned component
pipelines and publishes their exact rollout boundary, loss boundary, CPU stand-in, and result to the
GitHub Actions summary. Full model servers, distributed trainers, and accelerator tests remain
explicit environment-specific gates.

Security and compatibility guarantees
-------------------------------------

* Metadata is deterministic allowlisted JSON; pickle is never accepted.
* Tensor catalogs validate dtype, shape, size, layout, checksum, and buffer agreement before use.
* Configurable limits bound metadata, nesting, item count, tensor count, and aggregate allocation.
* Unknown schema types and unauthenticated payloads in authenticated mode fail closed.
* Framework-specific state stays in namespaced extensions instead of being silently discarded.
* Python API version ``0.x`` may evolve; the wire schema is independently versioned and validated.

License
-------

Apache License 2.0. See the repository license file for the full terms.
