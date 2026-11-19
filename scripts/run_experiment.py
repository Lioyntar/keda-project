"""
Τρέχει ένα ολόκληρο πείραμα autoscaling με μία εντολή:

    python scripts/run_experiment.py keda-combined
    python scripts/run_experiment.py keda-queue
    python scripts/run_experiment.py hpa-cpu
    python scripts/run_experiment.py keda-combined --scenario scenarios/smoke.json   (γρήγορη δοκιμή)

Προϋπόθεση: σε άλλο τερματικό τρέχει  kubectl port-forward svc/redis 6379:6379

Τι κάνει:
  1. Σβήνει όποιον autoscaler υπάρχει (ScaledObject / HPA) και φέρνει τους workers σε αρχική κατάσταση
     (0 για KEDA, 1 για HPA), με άδεια ουρά.
  2. Εφαρμόζει το k8s/autoscaling/<mode>.yaml και περιμένει να είναι έτοιμος ο autoscaler.
  3. Ξεκινά τον collector (κάθε 2 s) και στέλνει το σενάριο με τον generator.
  4. Συνεχίζει να μετράει λίγο μετά το τέλος (--tail) και σώζει όλα τα jobs.
Αποτελέσματα: results/<mode>_<ημερομηνία>/ με load.csv, samples.csv, jobs.csv, meta.json
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import redis
from kubernetes import client, config

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collector import Collector                     # noqa: E402
from generator import build_schedule, load_scenario, run_scenario   # noqa: E402

NS = "default"
MODES = {
    "keda-queue":    dict(file="k8s/autoscaling/keda-queue.yaml",    kind="keda", start_replicas=0),
    "keda-combined": dict(file="k8s/autoscaling/keda-combined.yaml", kind="keda", start_replicas=0),
    "hpa-cpu":       dict(file="k8s/autoscaling/hpa-cpu.yaml",       kind="hpa",  start_replicas=1),
}


def kubectl(*args, check=True):
    print("  $ kubectl " + " ".join(args))
    res = subprocess.run(["kubectl", *args], capture_output=True, text=True)
    if res.stdout.strip():
        print("    " + res.stdout.strip().replace("\n", "\n    "))
    if res.returncode != 0 and check:
        print("    " + res.stderr.strip())
        sys.exit(f"kubectl {' '.join(args)} failed")
    return res


def wait_for(cond, what, timeout=180, every=2):
    print(f"  ...waiting: {what}", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            print(f"  OK ({time.time() - t0:.0f}s)")
            return True
        print(".", end="", flush=True)
        time.sleep(every)
    print("  TIMEOUT")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--scenario", default="scenarios/main.json")
    ap.add_argument("--interval", type=float, default=2.0, help="collector sampling period (s)")
    ap.add_argument("--tail", type=int, default=60, help="seconds to keep measuring after the scenario")
    args = ap.parse_args()
    mode = MODES[args.mode]

    # --- 0. Έλεγχοι ---
    r = redis.Redis(host="localhost", port=6379)
    try:
        r.ping()
    except Exception:
        sys.exit("Δεν βρίσκω το Redis στο localhost:6379. Τρέχει το 'kubectl port-forward svc/redis 6379:6379';")
    config.load_kube_config()
    apps, core, custom = client.AppsV1Api(), client.CoreV1Api(), client.CustomObjectsApi()
    scenario = load_scenario(args.scenario)
    schedule, dt = build_schedule(scenario)
    print(f"Mode {args.mode}: scenario {args.scenario} "
          f"({len(schedule) * dt:.0f}s, {sum(s[3] for s in schedule)} jobs)")

    # --- 1. Καθαρή αρχή ---
    print("\n[1] Reset")
    kubectl("delete", "scaledobject", "--all", "-n", NS, "--ignore-not-found")
    kubectl("delete", "hpa", "--all", "-n", NS, "--ignore-not-found")
    time.sleep(3)
    r.delete("jobs", "done")
    n0 = mode["start_replicas"]
    kubectl("scale", "deployment/worker", f"--replicas={n0}", "-n", NS)

    def settled():
        pods = core.list_namespaced_pod(NS, label_selector="app=worker").items
        alive = [p for p in pods if p.metadata.deletion_timestamp is None]
        ready = apps.read_namespaced_deployment("worker", NS).status.ready_replicas or 0
        return len(pods) == n0 and len(alive) == n0 and ready == n0
    if not wait_for(settled, f"{n0} worker pod(s), no terminating pods"):
        sys.exit("Οι workers δεν ήρθαν στην αρχική κατάσταση.")

    # --- 2. Autoscaler ---
    print("\n[2] Apply autoscaler")
    kubectl("apply", "-f", mode["file"], "-n", NS)
    if mode["kind"] == "keda":
        def keda_ready():
            try:
                so = custom.get_namespaced_custom_object("keda.sh", "v1alpha1", NS, "scaledobjects", "worker-scaler")
            except Exception:
                return False
            conds = {c["type"]: c["status"] for c in so.get("status", {}).get("conditions", [])}
            return conds.get("Ready") == "True"
        if not wait_for(keda_ready, "ScaledObject Ready"):
            kubectl("describe", "scaledobject", "worker-scaler", "-n", NS, check=False)
            sys.exit("Το ScaledObject δεν έγινε Ready - δες το describe παραπάνω.")
    else:
        hpa_api = client.AutoscalingV2Api()

        def hpa_has_metrics():
            try:
                h = hpa_api.read_namespaced_horizontal_pod_autoscaler("worker-hpa", NS)
            except Exception:
                return False
            return bool(h.status.current_metrics)
        if not wait_for(hpa_has_metrics, "HPA reads CPU from metrics-server", timeout=240):
            sys.exit("Το HPA δεν παίρνει μετρήσεις CPU - τρέχει το metrics-server; (kubectl top pods)")
    print("  settling 20s...")
    time.sleep(20)

    # --- 3. Πείραμα ---
    run_dir = os.path.join("results", f"{args.mode}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir)
    shutil.copy(mode["file"], run_dir)
    shutil.copy(args.scenario, run_dir)
    meta = dict(mode=args.mode, autoscaler=mode["kind"], scenario=args.scenario,
                started_local=datetime.now().isoformat(timespec="seconds"),
                interval=args.interval, tail=args.tail, start_replicas=n0)

    print(f"\n[3] Running -> {run_dir}")
    col = Collector(os.path.join(run_dir, "samples.csv"), redis.Redis(host="localhost", port=6379),
                    interval=args.interval)
    col.start()
    time.sleep(args.interval * 2)                 # λίγα δείγματα πριν ξεκινήσει το φορτίο
    sent = run_scenario(scenario, r, "jobs", os.path.join(run_dir, "load.csv"), flush=False)
    print(f"  scenario done, {sent} jobs sent. Tail {args.tail}s...")
    time.sleep(args.tail)
    col.stop()
    col.join(timeout=10)

    # --- 4. Αποθήκευση ---
    raw = r.lrange("done", 0, -1)
    with open(os.path.join(run_dir, "jobs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "enq", "start", "end", "pod"])
        for item in raw:
            j = json.loads(item)
            w.writerow([j["id"], j["enq"], j["start"], j["end"], j["pod"]])
    left = r.llen("jobs")
    meta.update(jobs_sent=sent, jobs_done=len(raw), jobs_left_in_queue=left,
                collector_errors=col.errors)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[4] Saved {len(raw)} completed jobs ({left} still in queue).")
    if col.errors:
        print(f"  collector errors: {col.errors}")
    print(f"  Next: python scripts/analyze.py {run_dir}")


if __name__ == "__main__":
    main()
