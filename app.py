"""
app.py — Flask web server wrapping the Distributed File System
Ties together: foundation.py, replication.py, failure_detector.py, coordinator_demo.py

Run:
    pip install flask
    python app.py
Then open: http://localhost:5000
"""

import time
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, render_template

from foundation import (
    Config, NodeStatus, FileRecord, StorageNode,
    MetadataStore, NodeRegistry, EventLog, PerformanceReport,
)
from replication import ReplicationManager
from failure_detector import FailureDetector

app = Flask(__name__)

# ─── Coordinator (from coordinator_demo.py) ──────────────────────────────────

class Coordinator:
    def __init__(self, config=None):
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
        self._write_count       = 0
        self._start_time        = 0.0

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

    def add_node(self, node_id, capacity_mb):
        if self._registry.get(node_id):
            return False, "Node already exists"
        node = StorageNode(node_id=node_id, capacity_mb=capacity_mb)
        self._registry.register(node)
        self._failure_detector._ensure_node_state(node_id)
        self._start_heartbeat(node_id)
        self._log.log("COORDINATOR", f"Node {node_id} added ({capacity_mb}MB)")
        return True, "ok"

    def kill_node(self, node_id):
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

    def revive_node(self, node_id):
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

    def write_file(self, file_id, size_mb, checksum=""):
        if self._store.get_file(file_id):
            return False, f"File {file_id} already exists"
        t0 = time.perf_counter()
        record = FileRecord(file_id=file_id, size_mb=size_mb, checksum=checksum)
        self._store.add_file(record)
        candidates = sorted(
            [n for n in self._registry.online_nodes() if n.free_mb >= size_mb],
            key=lambda n: n.free_mb, reverse=True)
        seed = None
        if candidates:
            seed = candidates[0]
            self._store.add_replica(file_id, seed.node_id)
            self._registry.update_storage(seed.node_id, +size_mb)
            seed.local_files.add(file_id)
            self._log.log("WRITE", f"File {file_id} ({size_mb}MB) → {seed.node_id}")
        else:
            self._log.log("WARNING", f"No node has {size_mb}MB free for {file_id}")
        self._write_latencies.append(time.perf_counter() - t0)
        self._write_count += 1
        return True, seed.node_id if seed else None

    def read_file(self, file_id, requester=None):
        t0 = time.perf_counter()
        node_id = self._replication_mgr.route_read(file_id)
        if node_id:
            self._read_latencies.append(time.perf_counter() - t0 + 0.05)
            self._read_count += 1
            req_label = f" (requested by {requester})" if requester else ""
            self._log.log("ACCESS", f"Read {file_id} → served by {node_id}{req_label}")
            def _rel():
                time.sleep(0.05)
                self._replication_mgr.release_read(node_id)
            threading.Thread(target=_rel, daemon=True).start()
        return node_id, None

    def simulate_hot_access(self, file_id, count):
        if not self._store.get_file(file_id):
            return False, f"File {file_id} not found"
        for _ in range(count):
            self._store.record_access(file_id)
            self._read_count += 1
        self._log.log("ACCESS", f"Simulated {count} access(es) to {file_id}")
        return True, None

    def _on_node_failed(self, node_id):
        self._log.log("COORDINATOR", f"Coordinator notified: {node_id} FAILED")

    def _start_heartbeat(self, node_id):
        def _loop():
            while not self._heartbeat_stop.is_set():
                node = self._registry.get(node_id)
                if node is None:
                    break
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


# ─── Boot cluster ─────────────────────────────────────────────────────────────

coord = Coordinator()

for nid, cap in [("node1", 500), ("node2", 400), ("node3", 600), ("node4", 300)]:
    coord.add_node(nid, cap)

coord.start()
time.sleep(0.2)

for fid, size, cksum in [
    ("report_q1",   15,  "abc123"),
    ("report_q2",   20,  "def456"),
    ("archive_old", 50,  "ghi789"),
    ("config_main",  2,  "jkl012"),
    ("dataset_ml",  80,  "mno345"),
]:
    coord.write_file(fid, size, cksum)

coord.simulate_hot_access("report_q1",   60)
coord.simulate_hot_access("config_main", 30)
coord.simulate_hot_access("report_q2",   15)


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    tiers = coord.config.REPLICATION_TIERS
    nodes = []
    for n in sorted(coord.registry.all_nodes(), key=lambda x: x.node_id):
        nodes.append({
            "nodeId":     n.node_id,
            "status":     n.status.value,
            "capacityMB": n.capacity_mb,
            "usedMB":     round(n.used_mb, 1),
            "freeMB":     round(n.free_mb, 1),
            "currentLoad": n.current_load,
            "localFiles": sorted(n.local_files),
        })

    files = []
    for f in sorted(coord.store.all_files(), key=lambda x: -x.access_score):
        t = f.target_replicas(tiers)
        tier_name = "HOT" if t == 3 else "WARM" if t == 2 else "COLD"
        files.append({
            "fileId":       f.file_id,
            "sizeMB":       f.size_mb,
            "accessScore":  round(f.access_score, 2),
            "tier":         tier_name,
            "targetReplicas": t,
            "replicas":     sorted(f.replicas),
            "checksum":     f.checksum,
            "underReplicated": len(f.replicas) < t,
        })

    elapsed = time.time() - coord._start_time if coord._start_time else 1
    rl = coord._read_latencies
    wl = coord._write_latencies

    events = []
    for e in coord.log.all_events()[-40:]:
        events.append({
            "type":      e.event_type,
            "message":   e.message,
            "timestamp": e.timestamp,
        })
    events.reverse()

    ec = coord.log.count_by_type()
    avg_det = coord.log.average_detection_time()
    avg_rec = coord.log.average_recovery_time()

    return jsonify({
        "nodes": nodes,
        "files": files,
        "events": events,
        "metrics": {
            "readCount":    coord._read_count,
            "writeCount":   coord._write_count,
            "throughput":   round(coord._read_count / elapsed, 2),
            "avgReadMs":    round(sum(rl) / len(rl) * 1000, 1) if rl else 0,
            "avgWriteMs":   round(sum(wl) / len(wl) * 1000, 1) if wl else 0,
            "avgDetectionS": round(avg_det, 2) if avg_det else None,
            "avgRecoveryS":  round(avg_rec, 2) if avg_rec else None,
            "eventCounts":   ec,
            "storageEfficiency": round(coord.registry.storage_efficiency() * 100, 1),
            "uptime": round(elapsed, 0),
        }
    })


@app.route("/api/node/add", methods=["POST"])
def api_add_node():
    d = request.json
    ok, msg = coord.add_node(d["nodeId"], float(d["capacityMB"]))
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/node/kill", methods=["POST"])
def api_kill_node():
    d = request.json
    ok, msg = coord.kill_node(d["nodeId"])
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/node/revive", methods=["POST"])
def api_revive_node():
    d = request.json
    ok, msg = coord.revive_node(d["nodeId"])
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/file/write", methods=["POST"])
def api_write_file():
    d = request.json
    ok, seed = coord.write_file(d["fileId"], float(d["sizeMB"]), d.get("checksum", ""))
    return jsonify({"ok": ok, "seed": seed})


@app.route("/api/file/read", methods=["POST"])
def api_read_file():
    d = request.json
    node, err = coord.read_file(d["fileId"], requester=d.get("requestingNode"))
    return jsonify({"ok": node is not None, "node": node, "err": err})


@app.route("/api/file/burst", methods=["POST"])
def api_burst():
    d = request.json
    ok, err = coord.simulate_hot_access(d["fileId"], int(d["count"]))
    return jsonify({"ok": ok, "err": err})


if __name__ == "__main__":
    print("\n  DFS Dashboard → http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)