# replication.py
import time
import threading
from typing import Optional, List

from foundation import (
    Config, NodeStatus, NodeRegistry,
    MetadataStore, EventLog, FileRecord, StorageNode
)


class ReplicationManager:
    """
    Dynamic replication manager.

    Runs a single background loop that:
      1. Decays access scores first  — so old popularity fades over time
      2. Then checks every file      — compares current vs target replica count
                                       and adds/removes replicas to match demand

    Decay and replication always run in order within the same tick,
    preventing the race condition where replication acts on a stale score.

    Also provides load-balanced read routing:
      route_read(file_id) → picks the replica node with the lowest current load
    """

    def __init__(
        self,
        store: MetadataStore,
        registry: NodeRegistry,
        log: EventLog,
        config: Config,
    ):
        self._store    = store
        self._registry = registry
        self._log      = log
        self._config   = config

        # Single thread controls both decay and replication in guaranteed order
        self._stop_event = threading.Event()
        self._replication_thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────

    def start(self):
        """Start the single combined decay+replication loop as a daemon thread."""
        self._stop_event.clear()

        self._replication_thread = threading.Thread(
            target=self._replication_loop,
            name="ReplicationManager-Replication",
            daemon=True,
        )

        self._replication_thread.start()
        self._log.log("REPLICATION", "ReplicationManager started")

    def stop(self):
        """Gracefully stop the background thread."""
        self._stop_event.set()
        if self._replication_thread:
            self._replication_thread.join(timeout=5)
        self._log.log("REPLICATION", "ReplicationManager stopped")

    def route_read(self, file_id: str) -> Optional[str]:
        """
        Pick the best node to serve a read request for the given file.

        Strategy: among all online nodes that hold this file,
        return the one with the lowest current_load.
        Increments that node's load counter — caller must call
        release_read(node_id) when the read completes.

        Returns None if the file doesn't exist or has no online replicas.
        """
        record = self._store.get_file(file_id)
        if record is None or not record.replicas:
            self._log.log("WARNING", f"route_read: file {file_id} not found or has no replicas")
            return None

        # Find online nodes that hold this file
        candidates: List[StorageNode] = [
            n for n in self._registry.online_nodes()
            if n.node_id in record.replicas
        ]

        if not candidates:
            self._log.log("WARNING", f"route_read: no online replicas for {file_id}")
            return None

        # Pick node with lowest active load
        best = min(candidates, key=lambda n: n.current_load)

        # Record the access and bump load
        self._store.record_access(file_id)
        self._registry.update_load(best.node_id, +1)

        self._log.log(
            "ACCESS",
            f"Read of {file_id} routed to {best.node_id} "
            f"(load={best.current_load}, score={record.access_score:.2f})"
        )
        return best.node_id

    def release_read(self, node_id: str):
        """
        Call this when a read that was routed to node_id completes.
        Decrements the load counter so the node stays eligible for future reads.
        """
        self._registry.update_load(node_id, -1)

    # ──────────────────────────────────────────
    # COMBINED DECAY + REPLICATION LOOP
    # ──────────────────────────────────────────

    def _replication_loop(self):
        """
        Single loop that runs every REPLICATION_INTERVAL seconds.
        Decay always runs before the replication check — guaranteed order,
        no race between stale scores and replica decisions.
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(self._config.REPLICATION_INTERVAL)
            if self._stop_event.is_set():
                break

            # Step 1: decay first so scores reflect current reality
            self._store.decay_scores()
            self._log.log("REPLICATION", "Access scores decayed")

            # Step 2: then check files against their now-updated scores
            self._check_all_files()

    def _check_all_files(self):
        for record in self._store.all_files():
            target  = record.target_replicas(self._config.REPLICATION_TIERS)
            current = len(record.replicas)

            if current < target:
                self._add_replica(record, target)
            elif current > target:
                self._remove_replica(record, target)

    # ──────────────────────────────────────────
    # ADD REPLICA
    # ──────────────────────────────────────────

    def _add_replica(self, record: FileRecord, target: int):
        """
        Find the best node to place a new replica on and copy the file there.

        Placement rules:
          - Node must be ONLINE
          - Node must not already hold this file
          - Node must have enough free space for the file
          - Among valid candidates, pick the one with the best placement score:
              score = (free_mb / capacity_mb) * (1 - normalised_load)
            This balances free space against current read pressure.
        """
        candidates: List[StorageNode] = [
            n for n in self._registry.online_nodes()
            if n.node_id not in record.replicas
            and n.free_mb >= record.size_mb
        ]

        if not candidates:
            self._log.log(
                "WARNING",
                f"No candidate node found to add replica for {record.file_id} "
                f"(need {target} replicas, have {len(record.replicas)})"
            )
            return

        best = max(candidates, key=lambda n: self._placement_score(n))

        # Simulate the copy: update storage and replica set
        self._registry.update_storage(best.node_id, +record.size_mb)
        self._store.add_replica(record.file_id, best.node_id)
        best.local_files.add(record.file_id)

        self._log.log(
            "REPLICATION",
            f"Added replica of {record.file_id} → {best.node_id} "
            f"(score={record.access_score:.2f}, replicas: {len(record.replicas)-1} → {len(record.replicas)})"
        )

    # ──────────────────────────────────────────
    # REMOVE REPLICA
    # ──────────────────────────────────────────

    def _remove_replica(self, record: FileRecord, target: int):
        """
        Remove a replica from the most loaded node that holds this file.

        Rules:
          - Never drop below 1 replica (file must always exist somewhere)
          - Pick the node with the highest load — removing its replica
            reduces pressure and frees up its storage
          - Don't remove from a node that is the last surviving replica
        """
        if len(record.replicas) <= 1:
            return  # Safety guard — never delete the last copy

        # Among nodes holding this file, find the most loaded one
        holders: List[StorageNode] = [
            n for n in self._registry.online_nodes()
            if n.node_id in record.replicas
        ]

        if not holders:
            return

        # Most loaded = highest current_load, break ties by least free space
        worst = max(holders, key=lambda n: (n.current_load, -n.free_mb))

        # Simulate the removal
        self._registry.update_storage(worst.node_id, -record.size_mb)
        self._store.remove_replica(record.file_id, worst.node_id)
        worst.local_files.discard(record.file_id)

        self._log.log(
            "REPLICATION",
            f"Removed replica of {record.file_id} from {worst.node_id} "
            f"(score={record.access_score:.2f}, replicas: {len(record.replicas)+1} → {len(record.replicas)})"
        )

    # ──────────────────────────────────────────
    # PLACEMENT SCORE
    # ──────────────────────────────────────────

    def _placement_score(self, node: StorageNode) -> float:
        """
        Score a candidate node for replica placement.

        Formula: (free_mb / capacity_mb) * (1 - normalised_load)

        A node with lots of free space and low load scores high.
        A busy or nearly-full node scores low.

        current_load is treated as a raw integer count of active reads.
        We normalise it by assuming 10 concurrent reads = 100% load.
        Capped at 1.0 so the formula doesn't go negative.
        """
        capacity = node.capacity_mb if node.capacity_mb > 0 else 1
        space_ratio  = node.free_mb / capacity
        load_ratio   = min(node.current_load / 10.0, 1.0)
        return space_ratio * (1.0 - load_ratio)


# ──────────────────────────────────────────
# SMOKE TEST
# ──────────────────────────────────────────

if __name__ == "__main__":
    from foundation import FileRecord, StorageNode

    config   = Config()
    store    = MetadataStore(config)
    registry = NodeRegistry()
    log      = EventLog()

    # Register 4 nodes with different capacities
    for i, cap in enumerate([500, 300, 400, 600], start=1):
        registry.register(StorageNode(node_id=f"node{i}", capacity_mb=cap))

    # Add 3 files: hot, warm, cold
    store.add_file(FileRecord(file_id="file_hot",  size_mb=10, checksum="aaa"))
    store.add_file(FileRecord(file_id="file_warm", size_mb=20, checksum="bbb"))
    store.add_file(FileRecord(file_id="file_cold", size_mb=5,  checksum="ccc"))

    # Place one initial replica for each file on node1
    for fid in ["file_hot", "file_warm", "file_cold"]:
        store.add_replica(fid, "node1")
        registry.update_storage("node1", store.get_file(fid).size_mb)
        registry.get("node1").local_files.add(fid)

    # Simulate access: hot gets 60 reads, warm gets 12, cold gets 1
    for _ in range(60): store.record_access("file_hot")
    for _ in range(12): store.record_access("file_warm")
    store.record_access("file_cold")

    print("=== Starting ReplicationManager ===")
    manager = ReplicationManager(store, registry, log, config)
    manager.start()

    # Let the replication loop run for a couple of ticks
    time.sleep(config.REPLICATION_INTERVAL * 2 + 1)

    print("\n=== Replica State After Replication ===")
    for f in store.all_files():
        target = f.target_replicas(config.REPLICATION_TIERS)
        tier   = "HOT" if target == 3 else "WARM" if target == 2 else "COLD"
        print(f"  {f.file_id:<12} score={f.access_score:6.2f}  tier={tier}  "
              f"replicas={sorted(f.replicas)}  (target={target})")

    print("\n=== Node Storage After Replication ===")
    for n in sorted(registry.all_nodes(), key=lambda x: x.node_id):
        print(f"  {n.node_id}  used={n.used_mb:.1f}MB  free={n.free_mb:.1f}MB  "
              f"files={sorted(n.local_files)}")

    print("\n=== Route Read Test ===")
    # Simulate concurrent load on node1 to force routing elsewhere
    registry.update_load("node1", +5)
    routed_to = manager.route_read("file_hot")
    print(f"  file_hot read routed to: {routed_to}")
    if routed_to:
        manager.release_read(routed_to)

    print("\n=== Event Log ===")
    for e in log.all_events():
        print(f"  [{e.event_type}] {e.message}")

    manager.stop()