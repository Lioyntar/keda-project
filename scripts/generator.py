"""
Workload generator: στέλνει jobs στην ουρά Redis σύμφωνα με ένα σενάριο φάσεων.
Αφίξεις Poisson με σταθερό seed -> το ίδιο σενάριο δίνει πάντα το ίδιο φορτίο.
"""
import argparse
import csv
import json
import os
import time
import uuid
from datetime import datetime

import numpy as np
import redis


def target_rate(phase, t_in_phase):
    """Ρυθμός (jobs/s) σε μια στιγμή της φάσης: σταθερός ή γραμμική μετάβαση."""
    if "rate" in phase:
        return phase["rate"]
    frac = t_in_phase / phase["duration"]
    return phase["from"] + (phase["to"] - phase["from"]) * frac


def build_schedule(scenario):
    """Υπολογίζει εκ των προτέρων πόσα jobs στέλνονται σε κάθε χρονικό βήμα."""
    rng = np.random.default_rng(scenario["seed"])
    dt = scenario.get("tick_seconds", 0.25)
    schedule = []          # (t, όνομα φάσης, ρυθμός στόχος, πλήθος jobs)
    t = 0.0
    for phase in scenario["phases"]:
        steps = int(round(phase["duration"] / dt))
        for k in range(steps):
            rate = target_rate(phase, k * dt)
            n = int(rng.poisson(rate * dt))
            schedule.append((round(t, 3), phase["name"], rate, n))
            t += dt
    return schedule, dt


def plot_plan(schedule, dt, name):
    import matplotlib.pyplot as plt

    ts = [s[0] for s in schedule]
    plt.figure(figsize=(10, 4))
    plt.plot(ts, [s[3] / dt for s in schedule], alpha=0.35, label="generated (Poisson)")
    plt.plot(ts, [s[2] for s in schedule], linewidth=2, label="target rate")
    plt.xlabel("time (s)")
    plt.ylabel("jobs / s")
    plt.title(f"Workload scenario: {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"results/{name}_plan.png"
    plt.savefig(out, dpi=150)
    print(f"Plan saved to {out}")


def main():
    ap = argparse.ArgumentParser(description="Phase-based Redis workload generator")
    ap.add_argument("scenario", help="path to scenario JSON")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=6379)
    ap.add_argument("--queue", default="jobs")
    ap.add_argument("--dry-run", action="store_true", help="only compute and plot the plan")
    ap.add_argument("--no-flush", action="store_true", help="do not empty the queue first")
    args = ap.parse_args()

    with open(args.scenario) as f:
        scenario = json.load(f)
    name = os.path.splitext(os.path.basename(args.scenario))[0]
    schedule, dt = build_schedule(scenario)
    total = sum(s[3] for s in schedule)
    print(f"Scenario '{name}': {len(schedule) * dt:.0f}s, {total} jobs, seed={scenario['seed']}")

    os.makedirs("results", exist_ok=True)
    if args.dry_run:
        plot_plan(schedule, dt, name)
        return

    r = redis.Redis(host=args.host, port=args.port)
    if not args.no_flush:
        r.delete(args.queue)          # καθαρή αρχή για κάθε πείραμα

    out = f"results/{name}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    total_sent = 0
    start = time.time()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "unix_time", "phase", "target_rate", "sent", "total_sent", "queue_len"])
        try:
            for t, phase, rate, n in schedule:
                delay = start + t - time.time()    # κρατάμε σταθερό ρολόι
                if delay > 0:
                    time.sleep(delay)
                now = time.time()

                pipe = r.pipeline()                # όλα τα jobs του βήματος σε ένα round-trip
                for _ in range(n):
                    pipe.rpush(args.queue, json.dumps({"id": uuid.uuid4().hex, "enqueued_at": now}))
                pipe.llen(args.queue)
                queue_len = pipe.execute()[-1]

                total_sent += n
                w.writerow([t, f"{now:.3f}", phase, f"{rate:.2f}", n, total_sent, queue_len])
                print(f"\r t={t:6.1f}s  phase={phase:<10} rate={rate:5.1f}/s  "
                      f"sent={total_sent:6d}  queue={queue_len:6d}", end="", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted - partial log kept.")

    print(f"\nDone. {total_sent} jobs sent. Log: {out}")


if __name__ == "__main__":
    main()