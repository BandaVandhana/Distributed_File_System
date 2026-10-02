# failure_detector.py
import time
import queue
import threading
from collections import deque
from typing import Callable, Dict, Optional, List

from foundation import (
    Config, NodeStatus, NodeRegistry,
    MetadataStore, EventLog, StorageNode
)


# ──────────────────────────────────────────────────────────────
# OPTIMISATION 1: Phi Accrual — replaces hard timeout suspects
# ──────────────────────────────────────────────────────────────

class HeartbeatHistory:
    """
    Tracks the interval history of heartbeats for a single node.
    Used to compute a phi (φ) suspicion score instead of a hard timeout.

    How phi works:
      - φ = 0.0  → node is definitely alive
      - φ > PHI_SUSPECT (e.g. 1.0) → mark SUSPECTED
      - φ > PHI_FAIL   (e.g. 2.5) → mark FAILED

    The score is based on how much longer the node has been silent
    compared to its historical average heartbeat interval.
    A node that normally beats every 1s being silent for 3s scores
    much higher than a slow node that normally beats every 5s.
    """

    def __init__(self, max_samples: int = 200):
        self._intervals: deque = deque(maxlen=max_samples)
        self._last_beat: Optional[float] = None

    def record(self, now: float):
        """Call this every time a heartbeat arrives for this node."""
        if self._last_beat is not None:
            interval = now - self._last_beat
            if interval > 0:
                self._intervals.append(interval)
        self._last_beat = now

    def phi(self, now: float) -> float:
        """
        Returns the current suspicion level φ.
        Returns 0.0 if there is not enough history yet.
        """
        if len(self._intervals) < 2 or self._last_beat is None:
            return 0.0
        elapsed = now - self._last_beat
        mean = sum(self._intervals) / len(self._intervals)
        if mean <= 0:
            return 0.0
        # φ = elapsed / mean_interval
        # 1.0 = silent for exactly one average interval → suspicious
        # 2.5 = silent for 2.5x average interval → confirmed dead
        return elapsed / mean


# ──────────────────────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────────────────────

class FailureDetector:
    """
    Watchdog that monitors node heartbeats and handles:
      - ONLINE → SUSPECTED → FAILED transitions  (via Phi Accrual)
      - Re-replication of files on failed nodes   (rate-limited via queue)
      - Node rejoin / recovery reconciliation
      - Thread-safe status transitions             (via registry.update_status + per-node locks)
      - Smart replica placement                   (free space + load score)
    """

    def __init__(
        self,
        registry: NodeRegistry,
        store: MetadataStore,
        log: EventLog,
        config: Config,
        poll_interval: float = 2.0,
        replication_rate: float = 0.1,  # seconds between replication jobs (10/sec max)
        phi_suspect: float = 1.0,       # φ threshold to mark SUSPECTED
        phi_fail: float = 2.5,          # φ threshold to mark FAILED
    ):
        self._registry = registry
        self._store = store
        self._log = log
        self._config = config
        self._poll_interval = poll_interval
        self._replication_rate = replication_rate
        self._phi_suspect = phi_suspect
        self._phi_fail = phi_fail

        self._node_failed_callback: Optional[Callable[[str], None]] = None

        # Watchdog thread
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # OPTIMISATION 1: one HeartbeatHistory per node
        self._histories: Dict[str, HeartbeatHistory] = {}

        # OPTIMISATION 3: priority queue for rate-limited re-replication
        # Items: (priority, file_id, failed_node_id)
        # Lower priority number = more urgent (1 replica left = priority 1)
        self._replication_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._replication_thread: Optional[threading.Thread] = None

        # OPTIMISATION 4: per-node lock to guard status transitions
        # Prevents race between watchdog thread and notify_heartbeat caller
        self._node_locks: Dict[str, threading.Lock] = {}

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def on_node_failed(self, callback: Callable[[str], None]):
        """Register a callback the Coordinator uses to hear about failures."""
        self._node_failed_callback = callback

    def start(self):
        """Start the watchdog thread and the replication worker thread."""
        self._stop_event.clear()

        # Seed heartbeat history and locks for all already-registered nodes
        for node in self._registry.all_nodes():
            self._ensure_node_state(node.node_id)

        self._thread = threading.Thread(
            target=self._watchdog_loop,
            name="FailureDetector-Watchdog",
            daemon=True,
        )
        self._thread.start()

        # OPTIMISATION 3: separate thread drains replication queue
        self._replication_thread = threading.Thread(
            target=self._replication_worker,
            name="FailureDetector-Replication",
            daemon=True,
        )
        self._replication_thread.start()

        self._log.log(
            "DETECTOR",
            f"FailureDetector started (phi_suspect={self._phi_suspect}, phi_fail={self._phi_fail})"
        )

    def stop(self):
        """Gracefully stop both background threads."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._replication_thread:
            self._replication_thread.join(timeout=5)
        self._log.log("DETECTOR", "FailureDetector stopped")

    def notify_heartbeat(self, node_id: str):
        """
        Called by the Coordinator when a heartbeat arrives from a node.
        - Records the heartbeat in phi history
        - Updates last_heartbeat via registry
        - If node was SUSPECTED/FAILED, triggers rejoin logic
        """
        node = self._registry.get(node_id)
        if node is None:
            return

        self._ensure_node_state(node_id)

        now = time.time()

        # OPTIMISATION 1: record heartbeat in phi history
        self._histories[node_id].record(now)
        self._registry.update_heartbeat(node_id)

        # OPTIMISATION 4: acquire per-node lock before checking/changing status
        with self._node_locks[node_id]:
            if node.status in (NodeStatus.SUSPECTED, NodeStatus.FAILED):
                self._handle_node_rejoin(node)

    # ──────────────────────────────────────────
    # WATCHDOG LOOP
    # ──────────────────────────────────────────

    def _watchdog_loop(self):
        while not self._stop_event.is_set():
            self._check_all_nodes()
            self._stop_event.wait(self._poll_interval)

    def _check_all_nodes(self):
        now = time.time()
        for node in self._registry.all_nodes():
            self._ensure_node_state(node.node_id)
            self._evaluate_node(node, now)

    def _evaluate_node(self, node: StorageNode, now: float):
        """
        OPTIMISATION 1: Uses phi score instead of hard elapsed-time cutoff.
        Instead of 'if elapsed > 4s → FAILED', we ask 'how suspicious is
        the silence given this node's normal heartbeat pattern?'
        """
        # OPTIMISATION 4: lock per node so notify_heartbeat can't race us
        with self._node_locks[node.node_id]:
            phi = self._histories[node.node_id].phi(now)

            if node.status == NodeStatus.ONLINE:
                if phi >= self._phi_suspect:
                    self._transition_to_suspected(node, phi)

            elif node.status == NodeStatus.SUSPECTED:
                if phi >= self._phi_fail:
                    self._transition_to_failed(node, phi)

        # RECOVERING and FAILED nodes wait for a heartbeat (see notify_heartbeat)

    # ──────────────────────────────────────────
    # STATUS TRANSITIONS
    # ──────────────────────────────────────────

    def _transition_to_suspected(self, node: StorageNode, phi: float):
        """
        Called with node lock held.
        CHANGE 2: Uses registry.update_status() instead of node.status = directly.
        Keeps status changes inside the registry's own lock, consistent with
        how Person 1 designed the foundation.
        """
        self._registry.update_status(node.node_id, NodeStatus.SUSPECTED)
        self._log.log(
            "SUSPECTED",
            f"Node {node.node_id} marked SUSPECTED (φ={phi:.2f} ≥ {self._phi_suspect})",
        )

    def _transition_to_failed(self, node: StorageNode, phi: float):
        """
        Called with node lock held.
        CHANGE 2: Uses registry.update_status() instead of node.status = directly.
        """
        self._registry.update_status(node.node_id, NodeStatus.FAILED)
        self._log.log(
            "FAILED",
            f"Node {node.node_id} confirmed FAILED (φ={phi:.2f} ≥ {self._phi_fail})",
        )
        self._handle_node_failure(node.node_id)

    # ──────────────────────────────────────────
    # FAILURE HANDLING & RE-REPLICATION
    # ──────────────────────────────────────────

    def _handle_node_failure(self, node_id: str):
        """
        Find affected files and enqueue them for re-replication.
        OPTIMISATION 3: Jobs go into a priority queue; the worker drains
        them at a controlled rate to avoid flooding the network.
        CHANGE 3: Uses store.get_files_on_node() — new foundation method
        that does this lookup directly, replacing our manual filter loop.
        CHANGE 4: Uses store.remove_replica() — atomic method from foundation,
        replacing manual set math + update_replicas().
        """
        # CHANGE 3: foundation now provides get_files_on_node() directly
        affected = self._store.get_files_on_node(node_id)

        # Sort: fewest replicas first = most critical first
        affected.sort(key=lambda r: len(r.replicas))

        for record in affected:
            remaining = len(record.replicas) - 1  # -1 accounts for the node being removed

            urgency = "CRITICAL" if remaining <= 1 else "WARNING"
            self._log.log(
                urgency,
                f"File {record.file_id} lost replica on {node_id} — "
                f"{remaining} replica(s) remaining — queued for re-replication",
            )

            # CHANGE 4: remove_replica() is atomic — no manual set rebuilding
            self._store.remove_replica(record.file_id, node_id)

            # Priority = remaining replicas. Fewer = lower number = dequeued first
            self._replication_queue.put((remaining, record.file_id, node_id))

        # Fire the coordinator callback
        if self._node_failed_callback:
            self._node_failed_callback(node_id)

    # ──────────────────────────────────────────
    # OPTIMISATION 3: Rate-limited replication worker
    # ──────────────────────────────────────────

    def _replication_worker(self):
        """
        Background thread that drains the replication queue one job at a time,
        sleeping self._replication_rate seconds between jobs.
        Prevents a burst of node failures from flooding the network with
        simultaneous copy operations.
        """
        while not self._stop_event.is_set():
            try:
                priority, file_id, failed_node_id = self._replication_queue.get(timeout=1.0)
                self._do_replicate(file_id, failed_node_id)
                time.sleep(self._replication_rate)
            except queue.Empty:
                continue

    def _do_replicate(self, file_id: str, failed_node_id: str):
        """
        Perform one replication job from the queue.
        CHANGE 5: Uses store.add_replica() — atomic method from foundation,
        replacing manual replicas | {target} + update_replicas().
        Also calls registry.update_storage() to keep used_mb accurate.
        """
        record = self._store.get_file(file_id)
        if record is None:
            return  # file was deleted in the meantime

        target = self._pick_replication_target(failed_node_id, record.replicas)
        if target:
            self._log.log(
                "REPLICATION",
                f"Re-replicating {file_id} → {target}",
            )
            # CHANGE 5: add_replica() is the clean atomic method from new foundation
            self._store.add_replica(file_id, target)

            # Update storage accounting on the target node
            self._registry.update_storage(target, +record.size_mb)
        else:
            self._log.log(
                "WARNING",
                f"No available target to re-replicate {file_id}",
            )

    # ──────────────────────────────────────────
    # OPTIMISATION 2: Smarter replica placement
    # ──────────────────────────────────────────

    def _pick_replication_target(self, failed_node_id: str, existing_replicas) -> Optional[str]:
        """
        Pick the best available online node that doesn't already hold this file.

        OPTIMISATION 2: Scores candidates on both free space AND current load.

        CHANGE 6: current_load in new foundation is an int (active request count),
        NOT a 0.0–1.0 float. So we normalise it using: load_factor = load / (load + 1)
        This maps:
            0 requests → 0.0  (no penalty)
            1 request  → 0.5
            5 requests → 0.83
            ∞ requests → 1.0  (fully penalised)

        Score formula:
            space_ratio = free_mb / capacity_mb
            load_factor = current_load / (current_load + 1)
            score       = space_ratio * (1 - load_factor)
        """
        candidates: List[StorageNode] = [
            n for n in self._registry.online_nodes()
            if n.node_id != failed_node_id and n.node_id not in existing_replicas
        ]
        if not candidates:
            return None

        def score(n: StorageNode) -> float:
            capacity = n.capacity_mb or 1
            space_ratio = n.free_mb / capacity
            # CHANGE 6: normalise int load to 0–1 range
            load_factor = n.current_load / (n.current_load + 1)
            return space_ratio * (1.0 - load_factor)

        best = max(candidates, key=score)
        return best.node_id

    # ──────────────────────────────────────────
    # NODE REJOIN / RECOVERY
    # ──────────────────────────────────────────

    def _handle_node_rejoin(self, node: StorageNode):
        """
        Called (with node lock held) when a heartbeat arrives from a
        SUSPECTED or FAILED node. Reconcile its local file set then
        bring it back ONLINE.
        CHANGE 2: Uses registry.update_status() throughout.
        """
        self._registry.update_status(node.node_id, NodeStatus.RECOVERING)
        self._log.log(
            "RECOVERING",
            f"Node {node.node_id} is back — starting reconciliation",
        )
        self._reconcile_node(node)
        self._registry.update_status(node.node_id, NodeStatus.ONLINE)
        self._log.log(
            "ONLINE",
            f"Node {node.node_id} reconciled and back ONLINE",
        )

    def _reconcile_node(self, node: StorageNode):
        """
        Three-way reconcile:
        1. Files node claims to have but metadata doesn't know about → remove from node
        2. Files metadata says this node holds but node lost → remove from replica set
        3. Files node has that are valid → verify checksum FIRST, then re-add to replica set
            If checksum is missing or mismatched → quarantine the file, do NOT re-add

        SECURITY: Case 3 now performs checksum verification before accepting
        any file back. A node returning from failure cannot silently re-introduce
        corrupted or malicious data into the replica set.
        """
        meta_files = {r.file_id: r for r in self._store.all_files()}

        # Case 1: stale files on the node (not in metadata at all)
        stale = node.local_files - meta_files.keys()
        for fid in stale:
            node.local_files.discard(fid)
            self._log.log(
                "RECONCILE",
                f"Removed stale file {fid} from node {node.node_id}",
            )

        # Case 3: files node still has — VERIFY CHECKSUM before accepting back
        for fid in list(node.local_files):
            record = meta_files.get(fid)
            if not record:
                continue

            # ── Checksum gate ──────────────────────────────────────────
            node_checksum = self._fetch_node_checksum(node.node_id, fid)

            if not record.checksum:
                # No reference checksum stored — cannot verify, log a warning
                # but still accept (preserves backward-compat with old files)
                self._log.log(
                    "RECONCILE",
                    f"WARNING: {fid} on {node.node_id} has no reference checksum — "
                    f"accepted without verification (consider re-seeding checksum)",
                )
            elif node_checksum != record.checksum:
                # Mismatch — reject this replica, do NOT add it back
                node.local_files.discard(fid)
                self._log.log(
                    "INTEGRITY",
                    f"CHECKSUM MISMATCH: {fid} on {node.node_id} — "
                    f"expected={record.checksum!r} got={node_checksum!r} — "
                    f"replica REJECTED, queued for clean re-replication",
                )
                # Queue an emergency re-replication from a clean source
                self._replication_queue.put((1, fid, node.node_id))
                continue  # Do NOT add_replica — this copy is untrusted
            else:
                self._log.log(
                    "RECONCILE",
                    f"Checksum OK: {fid} on {node.node_id} verified ({record.checksum!r})",
                )
            # ──────────────────────────────────────────────────────────

            if node.node_id not in record.replicas:
                self._store.add_replica(fid, node.node_id)
                self._log.log(
                    "RECONCILE",
                    f"Re-added {node.node_id} as replica for {fid}",
                )

        # Case 2: files metadata says this node holds but node no longer has
        for fid, record in meta_files.items():
            if node.node_id in record.replicas and fid not in node.local_files:
                self._store.remove_replica(fid, node.node_id)
                self._log.log(
                    "RECONCILE",
                    f"Removed {node.node_id} from replica set of {fid} (file not present)",
                )

    def _fetch_node_checksum(self, node_id: str, file_id: str) -> str:
        record = self._store.get_file(file_id)
        if record is None:
                return ""
    # Simulated: pretend the node returns the same checksum it was seeded with.
    # In production: SSH/gRPC to node_id, compute sha256(file), return hex digest.
        return record.checksum

    # ──────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────

    def _ensure_node_state(self, node_id: str):
        """
        Lazily initialise HeartbeatHistory and Lock for a node the
        first time we see it. Safe to call multiple times.
        """
        if node_id not in self._histories:
            self._histories[node_id] = HeartbeatHistory()
            self._histories[node_id].record(time.time())
        if node_id not in self._node_locks:
            self._node_locks[node_id] = threading.Lock()


# ──────────────────────────────────────────
# QUICK SMOKE TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from foundation import FileRecord

    # CHANGE 1: MetadataStore now requires config as argument
    config = Config()
    registry = NodeRegistry()
    store = MetadataStore(config)   # <-- was MetadataStore() before, now needs config
    log = EventLog()

    # Two nodes
    n1 = StorageNode(node_id="node1", capacity_mb=500)
    n2 = StorageNode(node_id="node2", capacity_mb=500)
    registry.register(n1)
    registry.register(n2)

    # One file replicated on both
    f1 = FileRecord(file_id="file1", size_mb=10, replicas={"node1", "node2"})
    store.add_file(f1)

    def coordinator_callback(node_id):
        print(f"[COORDINATOR] Heard that {node_id} failed, taking action...")

    detector = FailureDetector(
        registry, store, log, config,
        poll_interval=1.0,
        replication_rate=0.05,
        phi_suspect=1.0,
        phi_fail=2.5,
    )
    detector.on_node_failed(coordinator_callback)
    detector.start()

    # Seed heartbeats to build phi history
    print("Seeding heartbeats to build phi history...")
    for _ in range(6):
        detector.notify_heartbeat("node1")
        detector.notify_heartbeat("node2")
        time.sleep(0.5)

    print("\nSimulating node1 going silent...")
    time.sleep(6)

    print("\n--- Event Log ---")
    for event in log.all_events():
        print(f"[{event.event_type}] {event.message}")

    print("\n--- File replicas after failure ---")
    print(store.get_file("file1").replicas)

    print("\nSimulating node1 rejoining...")
    n1.local_files = {"file1"}
    detector.notify_heartbeat("node1")
    time.sleep(0.5)

    print("\n--- Event Log (after rejoin) ---")
    for event in log.all_events():
        print(f"[{event.event_type}] {event.message}")

    detector.stop()