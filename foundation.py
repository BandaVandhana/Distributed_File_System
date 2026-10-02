from dataclasses import dataclass, field
from collections import deque
from threading import Lock
from typing import List, Dict, Optional, Set
from enum import Enum
import time


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

@dataclass
class Config:
    # Access score decays by this factor each interval (0.9 = 10% fade per tick)
    DECAY_FACTOR: float = 0.9

    # Heartbeat thresholds in seconds
    SUSPECT_TIMEOUT: float = 5.0    # missed heartbeats → SUSPECTED
    FAIL_TIMEOUT: float = 10.0      # missed heartbeats → FAILED

    # Replication tiers: (min_score_threshold, target_replica_count)
    # Checked top-down — first threshold the score exceeds wins
    # score >= 50  → 3 replicas (hot)
    # score >= 10  → 2 replicas (warm)
    # score >= 0   → 1 replica  (cold)
    REPLICATION_TIERS: list = field(default_factory=lambda: [
        (50, 3),
        (10, 2),
        (0,  1),
    ])

    MAX_REPLICAS: int = 3
    MAX_NODES: int = 10
    HEARTBEAT_INTERVAL: float = 2.0    # seconds between heartbeats
    REPLICATION_INTERVAL: float = 5.0  # seconds between replication manager ticks


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

class NodeStatus(Enum):
    ONLINE     = "ONLINE"
    SUSPECTED  = "SUSPECTED"
    FAILED     = "FAILED"
    RECOVERING = "RECOVERING"


# ─────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────

@dataclass
class FileRecord:
    file_id: str
    size_mb: float
    checksum: str = ""                               # MD5/SHA-256 for integrity checks
    replicas: Set[str] = field(default_factory=set)  # set of node_ids holding this file
    access_score: float = 0.0
    created_at: float = field(default_factory=time.time)

    def target_replicas(self, tiers: list) -> int:
        """Return target replica count based on current access score and tier config."""
        for threshold, count in tiers:
            if self.access_score >= threshold:
                return count
        return 1  # fallback: always keep at least 1


@dataclass
class StorageNode:
    node_id: str
    capacity_mb: float
    used_mb: float = 0.0
    status: NodeStatus = NodeStatus.ONLINE
    last_heartbeat: float = field(default_factory=time.time)
    current_load: int = 0                               # active read requests on this node
    local_files: Set[str] = field(default_factory=set)  # file_ids physically on this node

    @property
    def free_mb(self) -> float:
        """Available storage space in MB."""
        return self.capacity_mb - self.used_mb


@dataclass
class Event:
    timestamp: float
    event_type: str   # e.g. "REPLICATION", "FAILURE", "RECOVERY", "ACCESS"
    message: str


# ─────────────────────────────────────────
# EVENT LOG
# ─────────────────────────────────────────

class EventLog:
    """
    Thread-safe bounded log of system events.
    Used by ReplicationManager, FailureDetector, and PerformanceReport.
    """

    def __init__(self, maxlen: int = 500):
        self._log: deque = deque(maxlen=maxlen)
        self._lock = Lock()

    def log(self, event_type: str, message: str):
        """Append a new event. Thread-safe."""
        event = Event(
            timestamp=time.time(),
            event_type=event_type,
            message=message
        )
        with self._lock:
            self._log.append(event)

    def all_events(self) -> List[Event]:
        """Return snapshot of all events. Thread-safe."""
        with self._lock:
            return list(self._log)

    def events_by_type(self, event_type: str) -> List[Event]:
        """Return all events matching a given type. Thread-safe."""
        with self._lock:
            return [e for e in self._log if e.event_type == event_type]

    def count_by_type(self) -> Dict[str, int]:
        """Return a count of events grouped by event_type."""
        with self._lock:
            counts: Dict[str, int] = {}
            for e in self._log:
                counts[e.event_type] = counts.get(e.event_type, 0) + 1
            return counts

    def average_recovery_time(self) -> Optional[float]:
        """
        Compute mean time between paired FAILURE and RECOVERY events.
        Pairs them in order — first FAILURE with first RECOVERY, and so on.
        Returns None if no complete pairs exist.
        """
        with self._lock:
            failures  = [e.timestamp for e in self._log if e.event_type == "FAILURE"]
            recoveries = [e.timestamp for e in self._log if e.event_type == "RECOVERY"]
            pairs = list(zip(failures, recoveries))
            if not pairs:
                return None
            return sum(r - f for f, r in pairs) / len(pairs)

    def average_detection_time(self) -> Optional[float]:
        """
        Compute mean time between paired FAILURE and SUSPECTED events.
        Represents how fast the system detected a fault.
        Returns None if no complete pairs exist.
        """
        with self._lock:
            suspected  = [e.timestamp for e in self._log if e.event_type == "SUSPECTED"]
            failures   = [e.timestamp for e in self._log if e.event_type == "FAILURE"]
            pairs = list(zip(suspected, failures))
            if not pairs:
                return None
            return sum(f - s for s, f in pairs) / len(pairs)


# ─────────────────────────────────────────
# METADATA STORE
# ─────────────────────────────────────────

class MetadataStore:
    """
    Source of truth for all file metadata.
    All methods are thread-safe via an internal lock.
    Person 2 (ReplicationManager) and Person 3 (FailureDetector) both read/write here.
    """

    def __init__(self, config: Config):
        self._files: Dict[str, FileRecord] = {}
        self._lock = Lock()
        self._config = config

    def add_file(self, record: FileRecord):
        """Register a new file in the metadata store."""
        with self._lock:
            self._files[record.file_id] = record

    def get_file(self, file_id: str) -> Optional[FileRecord]:
        """Retrieve a FileRecord by ID. Returns None if not found."""
        with self._lock:
            return self._files.get(file_id)

    def all_files(self) -> List[FileRecord]:
        """Return a snapshot list of all FileRecords."""
        with self._lock:
            return list(self._files.values())

    def record_access(self, file_id: str):
        """Increment the access score for a file by 1. Called on every read."""
        with self._lock:
            if file_id in self._files:
                self._files[file_id].access_score += 1.0

    def decay_scores(self):
        """
        Apply exponential decay to all access scores.
        Uses DECAY_FACTOR from Config — called periodically by ReplicationManager.
        Formula: score = old_score * DECAY_FACTOR
        """
        with self._lock:
            for record in self._files.values():
                record.access_score *= self._config.DECAY_FACTOR

    def update_replicas(self, file_id: str, replicas: Set[str]):
        """Atomically replace the replica set for a file."""
        with self._lock:
            if file_id in self._files:
                self._files[file_id].replicas = replicas

    def get_files_on_node(self, node_id: str) -> List[FileRecord]:
        """Return all files that have a replica on the given node. Used by FailureDetector."""
        with self._lock:
            return [f for f in self._files.values() if node_id in f.replicas]

    def remove_replica(self, file_id: str, node_id: str):
        """Remove a single node from a file's replica set. Used after node failure."""
        with self._lock:
            if file_id in self._files:
                self._files[file_id].replicas.discard(node_id)

    def add_replica(self, file_id: str, node_id: str):
        """Add a single node to a file's replica set. Used after successful copy."""
        with self._lock:
            if file_id in self._files:
                self._files[file_id].replicas.add(node_id)

    def replica_distribution(self) -> Dict[int, int]:
        """
        Returns a dict mapping replica_count → number_of_files.
        e.g. {1: 5, 2: 3, 3: 8} means 5 files have 1 replica, 3 have 2, 8 have 3.
        Used by PerformanceReport.
        """
        with self._lock:
            dist: Dict[int, int] = {}
            for f in self._files.values():
                count = len(f.replicas)
                dist[count] = dist.get(count, 0) + 1
            return dist

    def under_replicated_files(self, config: Config) -> List[FileRecord]:
        """
        Return files where current replica count is below target.
        Used by PerformanceReport to measure replication health.
        """
        with self._lock:
            result = []
            for f in self._files.values():
                target = f.target_replicas(config.REPLICATION_TIERS)
                if len(f.replicas) < target:
                    result.append(f)
            return result


# ─────────────────────────────────────────
# NODE REGISTRY
# ─────────────────────────────────────────

class NodeRegistry:
    """
    Tracks all storage nodes and their current state.
    All methods are thread-safe via an internal lock.
    Person 2 uses this for placement/routing. Person 3 uses it for failure transitions.
    """

    def __init__(self):
        self._nodes: Dict[str, StorageNode] = {}
        self._lock = Lock()

    def register(self, node: StorageNode):
        """Add a new node to the registry."""
        with self._lock:
            self._nodes[node.node_id] = node

    def get(self, node_id: str) -> Optional[StorageNode]:
        """Retrieve a StorageNode by ID. Returns None if not found."""
        with self._lock:
            return self._nodes.get(node_id)

    def all_nodes(self) -> List[StorageNode]:
        """Return snapshot of all nodes regardless of status."""
        with self._lock:
            return list(self._nodes.values())

    def online_nodes(self) -> List[StorageNode]:
        """Return only nodes currently marked ONLINE. Used by ReplicationManager for placement."""
        with self._lock:
            return [n for n in self._nodes.values() if n.status == NodeStatus.ONLINE]

    def update_heartbeat(self, node_id: str):
        """Refresh the last_heartbeat timestamp for a node. Called when heartbeat is received."""
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_heartbeat = time.time()

    def update_status(self, node_id: str, status: NodeStatus):
        """
        Transition a node to a new status.
        Used by FailureDetector: ONLINE → SUSPECTED → FAILED → RECOVERING → ONLINE
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].status = status

    def update_load(self, node_id: str, delta: int):
        """
        Increment or decrement current_load on a node.
        Pass +1 when a read is routed to it, -1 when the read completes.
        Used by ReplicationManager for load-balanced routing.
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].current_load = max(0, self._nodes[node_id].current_load + delta)

    def update_storage(self, node_id: str, delta_mb: float):
        """
        Adjust used_mb on a node after a file copy or removal.
        Pass positive delta when adding a replica, negative when removing.
        """
        with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].used_mb = max(0.0, self._nodes[node_id].used_mb + delta_mb)

    def storage_efficiency(self) -> float:
        """
        Returns ratio of used storage to total capacity across all nodes (0.0 to 1.0).
        e.g. 0.45 means 45% of total cluster storage is in use.
        Used by PerformanceReport.
        """
        with self._lock:
            total = sum(n.capacity_mb for n in self._nodes.values())
            used  = sum(n.used_mb    for n in self._nodes.values())
            return round(used / total, 4) if total > 0 else 0.0

    def load_distribution(self) -> Dict[str, int]:
        """
        Returns a dict of node_id → current_load for all online nodes.
        Used by PerformanceReport to measure read distribution evenness.
        """
        with self._lock:
            return {
                n.node_id: n.current_load
                for n in self._nodes.values()
                if n.status == NodeStatus.ONLINE
            }

    def node_status_summary(self) -> Dict[str, int]:
        """
        Returns count of nodes in each status.
        e.g. {'ONLINE': 4, 'FAILED': 1, 'RECOVERING': 1}
        Used by PerformanceReport.
        """
        with self._lock:
            summary: Dict[str, int] = {}
            for n in self._nodes.values():
                key = n.status.value
                summary[key] = summary.get(key, 0) + 1
            return summary


# ─────────────────────────────────────────
# PERFORMANCE REPORT
# ─────────────────────────────────────────

class PerformanceReport:
    """
    Reads from MetadataStore, NodeRegistry, and EventLog to print
    a full system health and performance summary.
    Called by Person 4's Coordinator at the end of the simulation.
    """

    def __init__(self, store: MetadataStore, registry: NodeRegistry, log: EventLog, config: Config):
        self._store    = store
        self._registry = registry
        self._log      = log
        self._config   = config

    def print_report(self):
        divider = "=" * 55

        print(f"\n{divider}")
        print("   PERFORMANCE REPORT — Distributed File System")
        print(divider)

        # ── 1. Node Status ────────────────────────────────────
        print("\n[ NODE STATUS ]")
        status_summary = self._registry.node_status_summary()
        for status, count in sorted(status_summary.items()):
            print(f"  {status:<12} : {count} node(s)")

        all_nodes = self._registry.all_nodes()
        print(f"\n  {'Node':<10} {'Status':<12} {'Used MB':>8} {'Free MB':>8} {'Load':>6}")
        print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*6}")
        for n in sorted(all_nodes, key=lambda x: x.node_id):
            print(f"  {n.node_id:<10} {n.status.value:<12} {n.used_mb:>8.1f} {n.free_mb:>8.1f} {n.current_load:>6}")

        # ── 2. Storage Efficiency ─────────────────────────────
        print("\n[ STORAGE EFFICIENCY ]")
        efficiency = self._registry.storage_efficiency()
        total_cap  = sum(n.capacity_mb for n in all_nodes)
        total_used = sum(n.used_mb     for n in all_nodes)
        print(f"  Total capacity : {total_cap:.1f} MB")
        print(f"  Total used     : {total_used:.1f} MB")
        print(f"  Utilisation    : {efficiency * 100:.1f}%")

        # ── 3. Replica Distribution ───────────────────────────
        print("\n[ REPLICA DISTRIBUTION ]")
        dist = self._store.replica_distribution()
        all_files = self._store.all_files()
        total_files = len(all_files)
        print(f"  Total files    : {total_files}")
        for replica_count, file_count in sorted(dist.items()):
            bar = "█" * file_count
            print(f"  {replica_count} replica(s)  : {file_count} file(s)  {bar}")

        under = self._store.under_replicated_files(self._config)
        print(f"\n  Under-replicated files : {len(under)}")
        if under:
            for f in under:
                target = f.target_replicas(self._config.REPLICATION_TIERS)
                print(f"    - {f.file_id} : has {len(f.replicas)} replica(s), needs {target}")

        # ── 4. Access Score Summary ───────────────────────────
        print("\n[ ACCESS SCORES ]")
        if all_files:
            scores = sorted(all_files, key=lambda f: f.access_score, reverse=True)
            print(f"  {'File':<12} {'Score':>8} {'Replicas':>10} {'Tier':>6}")
            print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*6}")
            for f in scores:
                target = f.target_replicas(self._config.REPLICATION_TIERS)
                tier = "HOT" if target == 3 else "WARM" if target == 2 else "COLD"
                print(f"  {f.file_id:<12} {f.access_score:>8.2f} {len(f.replicas):>10} {tier:>6}")
        else:
            print("  No files in store.")

        # ── 5. Read Distribution ──────────────────────────────
        print("\n[ READ DISTRIBUTION (current load) ]")
        load_dist = self._registry.load_distribution()
        if load_dist:
            total_load = sum(load_dist.values())
            for node_id, load in sorted(load_dist.items()):
                pct = (load / total_load * 100) if total_load > 0 else 0
                bar = "█" * load
                print(f"  {node_id:<10} : {load:>3} requests  ({pct:.1f}%)  {bar}")
        else:
            print("  No online nodes.")

        # ── 6. Failure & Recovery Stats ───────────────────────
        print("\n[ FAILURE & RECOVERY ]")
        event_counts = self._log.count_by_type()
        failures     = event_counts.get("FAILURE",   0)
        recoveries   = event_counts.get("RECOVERY",  0)
        suspected    = event_counts.get("SUSPECTED", 0)
        replications = event_counts.get("REPLICATION", 0)

        print(f"  SUSPECTED events    : {suspected}")
        print(f"  FAILURE events      : {failures}")
        print(f"  RECOVERY events     : {recoveries}")
        print(f"  REPLICATION events  : {replications}")

        avg_recovery  = self._log.average_recovery_time()
        avg_detection = self._log.average_detection_time()

        if avg_detection is not None:
            print(f"\n  Avg detection time  : {avg_detection:.2f}s  (SUSPECTED → FAILED)")
        else:
            print(f"\n  Avg detection time  : N/A (no failure pairs logged)")

        if avg_recovery is not None:
            print(f"  Avg recovery time   : {avg_recovery:.2f}s  (FAILURE → RECOVERY)")
        else:
            print(f"  Avg recovery time   : N/A (no recovery pairs logged)")

        # ── 7. Event Log Tail ─────────────────────────────────
        print("\n[ LAST 10 EVENTS ]")
        all_events = self._log.all_events()
        recent = all_events[-10:] if len(all_events) >= 10 else all_events
        if recent:
            for e in recent:
                ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
                print(f"  [{ts}] {e.event_type:<14} {e.message}")
        else:
            print("  No events logged.")

        print(f"\n{divider}\n")


# ─────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    config = Config()

    store    = MetadataStore(config)
    registry = NodeRegistry()
    log      = EventLog()

    # Register nodes
    node1 = StorageNode(node_id="node1", capacity_mb=500)
    node2 = StorageNode(node_id="node2", capacity_mb=300)
    node3 = StorageNode(node_id="node3", capacity_mb=400)
    registry.register(node1)
    registry.register(node2)
    registry.register(node3)

    # Add files with checksums
    file1 = FileRecord(file_id="file1", size_mb=10, checksum="abc123")
    file2 = FileRecord(file_id="file2", size_mb=25, checksum="def456")
    file3 = FileRecord(file_id="file3", size_mb=5,  checksum="ghi789")
    store.add_file(file1)
    store.add_file(file2)
    store.add_file(file3)

    # Simulate access — file1 is hot, file2 is warm, file3 is cold
    for _ in range(60): store.record_access("file1")
    for _ in range(12): store.record_access("file2")
    store.record_access("file3")

    store.decay_scores()

    # Add replicas
    store.add_replica("file1", "node1")
    store.add_replica("file1", "node2")
    store.add_replica("file1", "node3")
    store.add_replica("file2", "node1")
    store.add_replica("file2", "node2")
    store.add_replica("file3", "node1")

    # Update storage usage
    registry.update_storage("node1", 40.0)
    registry.update_storage("node2", 35.0)
    registry.update_storage("node3", 10.0)

    # Simulate load on reads
    registry.update_load("node1", +3)
    registry.update_load("node2", +1)
    registry.update_load("node3", +2)

    # Simulate a failure and recovery cycle
    log.log("SUSPECTED", "node2 missed 3 heartbeats")
    time.sleep(0.05)
    log.log("FAILURE",   "node2 confirmed FAILED after timeout")
    registry.update_status("node2", NodeStatus.FAILED)
    store.remove_replica("file1", "node2")
    store.remove_replica("file2", "node2")
    log.log("REPLICATION", "file1 copied to node3 as recovery")
    log.log("REPLICATION", "file2 copied to node3 as recovery")
    time.sleep(0.05)
    log.log("RECOVERY",  "node2 back online after reconciliation")
    registry.update_status("node2", NodeStatus.RECOVERING)

    log.log("REPLICATION", "file3 replica added to node2")
    log.log("ACCESS",      "file1 read routed to node1 (lowest load)")

    # Print full report
    report = PerformanceReport(store, registry, log, config)
    report.print_report()