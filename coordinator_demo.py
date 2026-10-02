# coordinator_demo.py
"""
Interactive terminal demo for the Distributed File System.
Only coordinator-layer changes — foundation, replication, failure_detector unchanged.

Run:  python coordinator_demo.py
"""

import time
import threading
from typing import Optional

from foundation import (
    Config, NodeStatus,
    FileRecord, StorageNode,
    MetadataStore, NodeRegistry, EventLog,
    PerformanceReport,
)
from replication import ReplicationManager
from failure_detector import FailureDetector


# ─────────────────────────────────────────────────────────────
# ANSI COLOURS
# ─────────────────────────────────────────────────────────────

RESET  = "\033[0m";  BOLD = "\033[1m";  DIM = "\033[2m"
RED    = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
BLUE   = "\033[34m"; CYAN  = "\033[36m"; WHITE  = "\033[37m"
MAGENTA = "\033[35m"

def c(text, *codes): return "".join(codes) + str(text) + RESET

NODE_COLOR = {"node1": BLUE, "node2": CYAN, "node3": GREEN, "node4": YELLOW}
STATUS_STYLE = {
    NodeStatus.ONLINE:     (GREEN,  "●  ONLINE    "),
    NodeStatus.SUSPECTED:  (YELLOW, "◑  SUSPECTED "),
    NodeStatus.FAILED:     (RED,    "✗  FAILED    "),
    NodeStatus.RECOVERING: (CYAN,   "↻  RECOVERING"),
}
TIER_STYLE = {"HOT": (RED, "🔥 HOT"), "WARM": (YELLOW, "~ WARM"), "COLD": (BLUE, "* COLD")}

def ncolor(nid):        return NODE_COLOR.get(nid, WHITE)
def ok(m):   print(c("  ✔ ", GREEN)  + c(m, GREEN))
def warn(m): print(c("  ⚠ ", YELLOW) + c(m, YELLOW))
def fail(m): print(c("  ✗ ", RED)    + c(m, RED))
def info(m): print(c("    ", DIM)    + c(m, DIM))
def step(m): print(c("  ▸ ", CYAN)   + m)

def bar(used, total, w=18):
    if total == 0: return "[" + "?" * w + "]"
    f = int(used / total * w); pct = used / total
    col = RED if pct > 0.8 else YELLOW if pct > 0.6 else GREEN
    return c("[" + "█" * f + "░" * (w - f) + "]", col)

def div(title="", w=72):
    if title:
        p = (w - len(title) - 2) // 2
        return c("─" * p + " " + title + " " + "─" * (w - p - len(title) - 2), DIM)
    return c("─" * w, DIM)


# ─────────────────────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────────────────────

def print_cluster(coord):
    print("\n" + div("Cluster State"))
    all_nodes = coord.registry.all_nodes()
    all_files = coord.store.all_files()

    print(c(f"  {'NODE':<12} {'STATUS':<16} {'STORAGE':<30} FILES", BOLD))
    print(c("  " + "─" * 68, DIM))
    for n in sorted(all_nodes, key=lambda x: x.node_id):
        col, s = STATUS_STYLE.get(n.status, (WHITE, str(n.status)))
        nc = ncolor(n.node_id)
        files_here = [f.file_id for f in all_files if n.node_id in f.replicas]
        ftags = " ".join(c(fid, nc) for fid in files_here) or c("(empty)", DIM)
        print(f"  {c(n.node_id, BOLD+nc):<22} {c(s, col):<24} "
              f"{bar(n.used_mb, n.capacity_mb)}  {n.used_mb:.0f}/{n.capacity_mb}MB   {ftags}")

    tiers = coord.config.REPLICATION_TIERS
    print()
    print(c(f"  {'FILE':<15} {'SIZE':>6}  {'SCORE':>7}  {'TIER':<11} {'CHECKSUM':<16} REPLICAS", BOLD))
    print(c("  " + "─" * 80, DIM))
    for f in sorted(all_files, key=lambda x: -x.access_score):
        target = f.target_replicas(tiers)
        tname = "HOT" if target == 3 else "WARM" if target == 2 else "COLD"
        tcol, tstr = TIER_STYLE[tname]
        under = len(f.replicas) < target
        rdots = " ".join(c(f"[{nid}]", ncolor(nid)) for nid in sorted(f.replicas)) \
                or c("NONE — FILE LOST!", BOLD + RED)
        usub  = c(f" ⚠ needs {target}", BOLD + YELLOW) if under else ""
        scol  = RED if f.access_score >= 50 else YELLOW if f.access_score >= 10 else BLUE
        cksum = c(f.checksum[:14] if f.checksum else "—", DIM)
        print(f"  {c(f.file_id, BOLD):<23} {f.size_mb:>5.0f}MB  "
              f"{c(f'{f.access_score:6.1f}', scol)}  {c(tstr, tcol):<19} {cksum:<24} {rdots}{usub}")
    print(div())


def print_metrics(coord):
    print("\n" + div("Performance Metrics"))
    all_nodes = coord.registry.all_nodes()
    all_files = coord.store.all_files()
    tiers     = coord.config.REPLICATION_TIERS

    rl = coord._read_latencies
    print(c("  1. Read Latency", BOLD))
    if rl:
        avg = sum(rl) / len(rl) * 1000
        print(f"     avg={c(f'{avg:.1f}ms', GREEN)}  min={c(f'{min(rl)*1000:.1f}ms', CYAN)}  "
              f"max={c(f'{max(rl)*1000:.1f}ms', YELLOW)}  samples={len(rl)}")
    else:
        print(c("     — no reads yet", DIM))

    wl = coord._write_latencies
    print(c("  2. Write Latency", BOLD))
    if wl:
        print(f"     avg={c(f'{sum(wl)/len(wl)*1000:.1f}ms', GREEN)}  samples={len(wl)}")
    else:
        print(c("     — no writes yet", DIM))

    elapsed = time.time() - coord._start_time if coord._start_time else 1
    tput = coord._read_count / elapsed
    tbar = "█" * min(int(tput * 2), 30)
    tcol = GREEN if tput > 5 else YELLOW if tput > 1 else BLUE
    print(c("  3. System Throughput", BOLD))
    print(f"     {c(f'{tput:.2f} req/s', tcol)}  {c(tbar, tcol)}  "
          f"({coord._read_count} reads in {elapsed:.1f}s)")

    avg_det = coord.log.average_detection_time()
    print(c("  4. Failure Detection Time", BOLD))
    if avg_det is not None:
        dc = GREEN if avg_det < 3 else YELLOW if avg_det < 7 else RED
        print(f"     {c(f'{avg_det:.2f}s', dc)}  (phi accrual: SUSPECTED → FAILED)")
    else:
        print(c("     — no failure pairs yet", DIM))

    avg_rec = coord.log.average_recovery_time()
    print(c("  5. Replica Recovery Time", BOLD))
    if avg_rec is not None:
        rc = GREEN if avg_rec < 5 else YELLOW if avg_rec < 10 else RED
        print(f"     {c(f'{avg_rec:.2f}s', rc)}  (FAILURE → RECOVERY)")
    else:
        print(c("     — no recovery pairs yet", DIM))

    online_ids = {n.node_id for n in all_nodes if n.status == NodeStatus.ONLINE}
    reachable  = sum(1 for f in all_files if f.replicas & online_ids)
    total      = len(all_files)
    pct        = reachable / total * 100 if total else 100
    pcol       = GREEN if pct == 100 else YELLOW if pct >= 60 else RED
    abar       = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    under      = [f for f in all_files if len(f.replicas) < f.target_replicas(tiers)]
    print(c("  6. Data Availability", BOLD))
    print(f"     {c(f'{pct:.1f}%', BOLD+pcol)}  [{c(abar, pcol)}]  {reachable}/{total} files reachable")
    if under:
        for f in under:
            warn(f"  {f.file_id}: {len(f.replicas)} replica(s), needs {f.target_replicas(tiers)}")
    else:
        ok("  All files at target replica count")
    print(div())


# ─────────────────────────────────────────────────────────────
# COORDINATOR
# ─────────────────────────────────────────────────────────────

class Coordinator:
    def __init__(self, config: Optional[Config] = None):
        self._config   = config or Config()
        self._store    = MetadataStore(self._config)
        self._registry = NodeRegistry()
        self._log      = EventLog()

        self._replication_mgr = ReplicationManager(
            store=self._store, registry=self._registry,
            log=self._log, config=self._config)
        self._failure_detector = FailureDetector(
            registry=self._registry, store=self._store,
            log=self._log, config=self._config,
            poll_interval=1.0, replication_rate=0.1,
            phi_suspect=1.0, phi_fail=2.5)
        self._failure_detector.on_node_failed(self._on_node_failed)

        self._heartbeat_stop    = threading.Event()
        self._heartbeat_threads = []
        self._read_latencies    = []
        self._write_latencies   = []
        self._read_count        = 0
        self._start_time        = 0.0

    # ── lifecycle ─────────────────────────────────────────────

    def start(self):
        self._start_time = time.time()
        self._heartbeat_stop.clear()
        self._replication_mgr.start()
        self._failure_detector.start()
        self._log.log("COORDINATOR", "Coordinator started")

    def stop(self):
        self._heartbeat_stop.set()
        for t in self._heartbeat_threads:
            t.join(timeout=3)
        self._replication_mgr.stop()
        self._failure_detector.stop()
        self._log.log("COORDINATOR", "Coordinator stopped")

    # ── node management ───────────────────────────────────────

    def add_node(self, node_id: str, capacity_mb: float):
        if self._registry.get(node_id):
            return False
        node = StorageNode(node_id=node_id, capacity_mb=capacity_mb)
        self._registry.register(node)
        self._failure_detector._ensure_node_state(node_id)
        self._start_heartbeat(node_id)
        self._log.log("COORDINATOR", f"Node {node_id} added ({capacity_mb} MB)")
        return True

    def kill_node(self, node_id: str):
        node = self._registry.get(node_id)
        if node is None:
            return False, f"Node {node_id} not found"
        if node.status == NodeStatus.FAILED:
            return False, f"Node {node_id} is already FAILED"
        self._registry.update_status(node_id, NodeStatus.FAILED)
        self._log.log("FAILURE", f"Node {node_id} forcibly killed")
        affected = self._store.get_files_on_node(node_id)
        for record in sorted(affected, key=lambda r: len(r.replicas)):
            self._store.remove_replica(record.file_id, node_id)
            priority = 1 if len(record.replicas) == 0 else len(record.replicas)
            self._failure_detector._replication_queue.put(
                (priority, record.file_id, node_id))
        self._log.log("FAILURE", f"{len(affected)} file(s) affected by {node_id} failure")
        return True, f"{len(affected)} file(s) affected"

    def revive_node(self, node_id: str):
        node = self._registry.get(node_id)
        if node is None:
            return False, f"Node {node_id} not found"
        if node.status != NodeStatus.FAILED:
            return False, f"Node {node_id} is not FAILED (currently {node.status.value})"
        node.status = NodeStatus.ONLINE
        self._registry.update_heartbeat(node_id)
        self._failure_detector.notify_heartbeat(node_id)
        self._start_heartbeat(node_id)
        self._log.log("COORDINATOR", f"Node {node_id} revived")
        return True, "ok"

    # ── file operations ───────────────────────────────────────

    def write_file(self, file_id: str, size_mb: float, checksum: str = ""):
        if self._store.get_file(file_id):
            return False, f"File {file_id} already exists"
        t0 = time.perf_counter()
        record = FileRecord(file_id=file_id, size_mb=size_mb, checksum=checksum)
        self._store.add_file(record)
        candidates = sorted(
            [n for n in self._registry.online_nodes() if n.free_mb >= size_mb],
            key=lambda n: n.free_mb, reverse=True)
        if candidates:
            seed = candidates[0]
            self._store.add_replica(file_id, seed.node_id)
            self._registry.update_storage(seed.node_id, +size_mb)
            seed.local_files.add(file_id)
            self._log.log("WRITE", f"File {file_id} ({size_mb}MB) → {seed.node_id}")
        else:
            self._log.log("WARNING", f"No node has {size_mb}MB free for {file_id}")
        self._write_latencies.append(time.perf_counter() - t0)
        return True, candidates[0].node_id if candidates else None

    def read_file(self, file_id: str, prefer_node: str = None):
        """
        Route a read for file_id.
        prefer_node: if set and valid, serve from that node directly.
                     Otherwise fall back to load-balanced routing.
        This is how 'node A requests a file to be served by node B' works:
          - prefer_node = B (the serving/target node)
        """
        t0 = time.perf_counter()

        if prefer_node:
            record = self._store.get_file(file_id)
            pnode  = self._registry.get(prefer_node)
            if (record and pnode
                    and pnode.status == NodeStatus.ONLINE
                    and prefer_node in record.replicas):
                self._store.record_access(file_id)
                self._registry.update_load(prefer_node, +1)
                self._read_count += 1
                self._read_latencies.append(time.perf_counter() - t0 + 0.05)
                self._log.log("ACCESS",
                    f"Read {file_id} explicitly served by {prefer_node} (manual request)")
                def _rel():
                    time.sleep(0.05)
                    self._replication_mgr.release_read(prefer_node)
                threading.Thread(target=_rel, daemon=True).start()
                return prefer_node, None
            else:
                # Diagnose why we can't honour the request
                if record is None:
                    reason = f"file '{file_id}' not found"
                elif pnode is None:
                    reason = f"node '{prefer_node}' not in cluster"
                elif pnode.status != NodeStatus.ONLINE:
                    reason = f"{prefer_node} is {pnode.status.value}, not ONLINE"
                elif prefer_node not in record.replicas:
                    reason = f"{prefer_node} does not hold a replica of '{file_id}'"
                else:
                    reason = "unknown"
                return None, reason

        # Load-balanced fallback
        node_id = self._replication_mgr.route_read(file_id)
        if node_id:
            self._read_latencies.append(time.perf_counter() - t0 + 0.05)
            self._read_count += 1
            def _rel():
                time.sleep(0.05)
                self._replication_mgr.release_read(node_id)
            threading.Thread(target=_rel, daemon=True).start()
        return node_id, None

    def simulate_hot_access(self, file_id: str, count: int):
        if not self._store.get_file(file_id):
            return False, f"File {file_id} not found"
        for _ in range(count):
            self._store.record_access(file_id)
            self._read_count += 1
        self._log.log("ACCESS", f"Simulated {count} access(es) to {file_id}")
        return True, None

    def print_report(self):
        PerformanceReport(
            self._store, self._registry, self._log, self._config
        ).print_report()

    # ── internals ─────────────────────────────────────────────

    def _on_node_failed(self, node_id):
        self._log.log("COORDINATOR", f"Coordinator notified: {node_id} FAILED")

    def _start_heartbeat(self, node_id):
        def _loop():
            while not self._heartbeat_stop.is_set():
                node = self._registry.get(node_id)
                if node is None: break
                if node.status not in (NodeStatus.FAILED,):
                    self._failure_detector.notify_heartbeat(node_id)
                self._heartbeat_stop.wait(self._config.HEARTBEAT_INTERVAL)
        t = threading.Thread(target=_loop, name=f"Heartbeat-{node_id}", daemon=True)
        t.start()
        self._heartbeat_threads.append(t)

    @property
    def store(self):    return self._store
    @property
    def registry(self): return self._registry
    @property
    def log(self):      return self._log
    @property
    def config(self):   return self._config


# ─────────────────────────────────────────────────────────────
# MENU + HELPERS
# ─────────────────────────────────────────────────────────────

MENU = """
{line}
  {b}Distributed File System — Interactive Demo{r}
{line}
  {cy}[s]{r} Show cluster state         {cy}[m]{r} Show performance metrics
  {cy}[w]{r} Write a file               {cy}[r]{r} Read file (load-balanced)
  {cy}[R]{r} Read from a specific node  {cy}[a]{r} Simulate access burst
  {cy}[n]{r} Add a new node             {cy}[k]{r} Kill a node
  {cy}[v]{r} Revive a failed node       {cy}[l]{r} Show event log (last 20)
  {cy}[i]{r} Show integrity events      {cy}[p]{r} Full performance report
  {cy}[q]{r} Quit
{line}"""

def show_menu():
    print(MENU.format(line=c("─" * 52, DIM), b=BOLD, r=RESET, cy=CYAN))

def prompt(text):
    try:
        return input(c(f"  › {text}: ", CYAN)).strip()
    except (EOFError, KeyboardInterrupt):
        return ""

def _node_ids(coord):
    return sorted(n.node_id for n in coord.registry.all_nodes())

def _file_ids(coord):
    return sorted(f.file_id for f in coord.store.all_files())


# ─────────────────────────────────────────────────────────────
# ACTION HANDLERS
# ─────────────────────────────────────────────────────────────

def handle_write(coord):
    print(c("\n  — Write a file —", BOLD))
    fid = prompt("File ID (e.g. report_q3)")
    if not fid: return
    if coord.store.get_file(fid):
        warn(f"File '{fid}' already exists"); return
    try:
        size = float(prompt("Size in MB (e.g. 25)"))
    except ValueError:
        warn("Invalid size — enter a number"); return
    cksum = prompt("Checksum (press Enter to skip)") or ""
    ok_flag, seed = coord.write_file(fid, size, cksum)
    if ok_flag:
        if seed:
            ok(f"'{fid}' ({size}MB) written — initial replica on {c(seed, BOLD+ncolor(seed))}")
        else:
            warn(f"'{fid}' registered but no node had {size}MB free for initial replica")
    else:
        fail(seed)


def handle_read_lb(coord):
    print(c("\n  — Read file (load-balanced) —", BOLD))
    fids = _file_ids(coord)
    if not fids: warn("No files in system yet"); return
    info("Available: " + ", ".join(fids))
    fid = prompt("File ID")
    if not fid: return
    node, err = coord.read_file(fid)
    if node:
        ok(f"'{fid}' → routed to {c(node, BOLD+ncolor(node))}")
    else:
        fail(f"No online replica for '{fid}'" + (f" ({err})" if err else ""))


def handle_read_explicit(coord):
    """
    Simulates a directed read: one node requests a file,
    specifying which node should serve it.
    e.g. 'node2 requests report_q1, to be served by node4'
    """
    print(c("\n  — Read from a specific node —", BOLD))
    info("Simulates: requesting node sends a read,")
    info("           target node serves the file.")
    info("Example: 'node2 requests report_q1 from node4'")
    print()

    fids = _file_ids(coord)
    if not fids: warn("No files in system yet"); return
    info("Files: " + ", ".join(fids))
    fid = prompt("File ID to read")
    if not fid or fid not in fids:
        warn("File not found"); return

    # Show which nodes hold replicas right now
    record = coord.store.get_file(fid)
    if record and record.replicas:
        holders = sorted(record.replicas)
        print(c(f"    Replicas on: " +
                " ".join(c(n, BOLD+ncolor(n)) for n in holders), DIM))

    nids = _node_ids(coord)
    info("Nodes: " + ", ".join(nids))
    src = prompt("Requesting node (who wants the data, e.g. node2)")
    if not src: return
    tgt = prompt("Target / serving node  (who should serve it, e.g. node4)")
    if not tgt: return

    step(f"{c(src, BOLD+ncolor(src))} → '{fid}' → served by {c(tgt, BOLD+ncolor(tgt))}")
    node, err = coord.read_file(fid, prefer_node=tgt)

    if node == tgt:
        ok(f"'{fid}' served by {c(tgt, BOLD+ncolor(tgt))} as requested")
    elif node:
        warn(f"Could not use {tgt} ({err}) — fell back to {c(node, BOLD+ncolor(node))}")
    else:
        fail(f"No online replica available for '{fid}'" + (f": {err}" if err else ""))


def handle_access_burst(coord):
    print(c("\n  — Simulate access burst —", BOLD))
    info("Raises access score → may promote tier and add replicas")
    info("HOT (score ≥ 50) → 3 replicas  |  WARM (≥ 10) → 2  |  COLD → 1")
    fids = _file_ids(coord)
    if not fids: warn("No files yet"); return
    info("Files: " + ", ".join(fids))
    fid = prompt("File ID")
    if not fid or fid not in fids:
        warn("File not found"); return
    try:
        count = int(prompt("Number of reads to simulate"))
    except ValueError:
        warn("Enter a whole number"); return
    before = coord.store.get_file(fid).access_score
    coord.simulate_hot_access(fid, count)
    f2     = coord.store.get_file(fid)
    target = f2.target_replicas(coord.config.REPLICATION_TIERS)
    tname  = "HOT" if target == 3 else "WARM" if target == 2 else "COLD"
    tcol   = TIER_STYLE[tname][0]
    ok(f"Score: {before:.1f} → {c(f'{f2.access_score:.1f}', RED)}  "
       f"Tier: {c(tname, tcol)}  Target replicas: {target}")
    info("Replication manager acts on next tick (~5s)")


def handle_add_node(coord):
    print(c("\n  — Add a new node —", BOLD))
    nid = prompt("Node ID (e.g. node5)")
    if not nid: return
    if coord.registry.get(nid):
        warn(f"Node '{nid}' already exists"); return
    try:
        cap = float(prompt("Capacity in MB (e.g. 500)"))
    except ValueError:
        warn("Invalid capacity"); return
    added = coord.add_node(nid, cap)
    if added:
        ok(f"Node {c(nid, BOLD+ncolor(nid))} joined the cluster ({cap}MB)")
        info("Heartbeat started — node immediately available for replication")


def handle_kill(coord):
    print(c("\n  — Kill a node —", BOLD))
    online = [n for n in coord.registry.all_nodes() if n.status == NodeStatus.ONLINE]
    if not online: warn("No ONLINE nodes to kill"); return
    info("Online: " + ", ".join(c(n.node_id, ncolor(n.node_id))
                                for n in sorted(online, key=lambda x: x.node_id)))
    nid = prompt("Node ID to kill")
    if not nid: return
    ok_flag, msg = coord.kill_node(nid)
    if ok_flag:
        fail(f"Node {c(nid, ncolor(nid))} KILLED — {msg}")
        info("Re-replication queued. Press [s] in ~5s to see self-healing.")
    else:
        warn(msg)


def handle_revive(coord):
    print(c("\n  — Revive a failed node —", BOLD))
    failed = [n for n in coord.registry.all_nodes() if n.status == NodeStatus.FAILED]
    if not failed: warn("No FAILED nodes to revive"); return
    info("Failed: " + ", ".join(c(n.node_id, RED) for n in failed))
    nid = prompt("Node ID to revive")
    if not nid: return
    ok_flag, msg = coord.revive_node(nid)
    if ok_flag:
        ok(f"Node {c(nid, ncolor(nid))} revived — reconciliation running")
        info("Node will be ONLINE within a couple of seconds")
    else:
        warn(msg)


def handle_log(coord):
    print("\n" + div("Event Log (last 20)"))
    TYPE_COL = {
        "FAILURE": RED, "RECOVERY": GREEN, "REPLICATION": BLUE,
        "ACCESS": MAGENTA, "WARNING": YELLOW, "WRITE": CYAN,
        "INTEGRITY": RED, "RECONCILE": CYAN, "ONLINE": GREEN,
        "SUSPECTED": YELLOW, "RECOVERING": CYAN,
    }
    events = coord.log.all_events()
    recent = events[-20:] if len(events) >= 20 else events
    for e in reversed(recent):
        ts  = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
        col = TYPE_COL.get(e.event_type, WHITE)
        print(f"  {c(ts, DIM)}  {c(f'{e.event_type:<14}', BOLD+col)}  {e.message}")
    print(div())


def handle_integrity(coord):
    """Show all INTEGRITY and RECONCILE events — checksum verification results."""
    print("\n" + div("Integrity & Reconciliation Events"))
    events = coord.log.all_events()
    integrity_types = {"INTEGRITY", "RECONCILE", "RECOVERING", "ONLINE"}
    filtered = [e for e in events if e.event_type in integrity_types]
    if not filtered:
        info("No integrity or reconciliation events yet.")
        info("Kill a node then revive it to trigger reconciliation.")
    else:
        TYPE_COL = {
            "INTEGRITY":  RED,
            "RECONCILE":  CYAN,
            "RECOVERING": MAGENTA,
            "ONLINE":     GREEN,
        }
        for e in filtered:
            ts  = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            col = TYPE_COL.get(e.event_type, WHITE)
            print(f"  {c(ts, DIM)}  {c(f'{e.event_type:<12}', BOLD+col)}  {e.message}")
    print(div())


# ─────────────────────────────────────────────────────────────
# BOOT + MAIN LOOP
# ─────────────────────────────────────────────────────────────

def boot_cluster():
    print(c("\n╔" + "═"*52 + "╗", BOLD+CYAN))
    print(c("║   Distributed File System  —  Interactive Demo   ║", BOLD+CYAN))
    print(c("╚" + "═"*52 + "╝", BOLD+CYAN))

    step("Starting 4-node cluster with 5 pre-loaded files...")
    coord = Coordinator()
    for nid, cap in [("node1", 500), ("node2", 400), ("node3", 600), ("node4", 300)]:
        coord.add_node(nid, cap)
        ok(f"  {c(nid, BOLD+ncolor(nid))} registered ({cap}MB)")

    coord.start()
    time.sleep(0.3)

    print()
    for fid, size, cksum in [
        ("report_q1",   15,  "abc"),
        ("report_q2",   20,  "def"),
        ("archive_old", 50,  "ghi"),
        ("config_main",  2,  "jkl"),
        ("dataset_ml",  100, "mno"),
    ]:
        _, seed = coord.write_file(fid, size, cksum)
        ok(f"  {fid:<14} ({size:>3}MB) → seeded on "
           f"{c(seed, ncolor(seed)) if seed else c('none', RED)}")

    # Pre-warm scores so tiers are interesting from the start
    coord.simulate_hot_access("report_q1",   60)
    coord.simulate_hot_access("config_main", 30)
    coord.simulate_hot_access("report_q2",   15)
    print()
    info("Scores pre-warmed: report_q1=HOT(60), config_main=HOT(30), report_q2=WARM(15)")
    info("Replication runs every 5s in background — replicas spread automatically")
    return coord


DISPATCH = {
    's': lambda coord: print_cluster(coord),
    'm': lambda coord: print_metrics(coord),
    'w': handle_write,
    'r': handle_read_lb,
    'R': handle_read_explicit,
    'a': handle_access_burst,
    'n': handle_add_node,
    'k': handle_kill,
    'v': handle_revive,
    'l': handle_log,
    'i': handle_integrity,
    'p': lambda coord: coord.print_report(),
}

if __name__ == "__main__":
    coord = boot_cluster()
    print_cluster(coord)
    show_menu()

    while True:
        try:
            choice = input(c("  › Command (? for menu): ", BOLD+CYAN)).strip()
        except (EOFError, KeyboardInterrupt):
            choice = "q"

        if choice == "q":
            step("Shutting down...")
            coord.stop()
            ok("Cluster stopped. Goodbye.")
            break
        elif choice in ("?", "h", "help", "menu", ""):
            show_menu()
        elif choice in DISPATCH:
            DISPATCH[choice](coord)
        else:
            warn(f"Unknown command '{choice}' — type ? for the menu")