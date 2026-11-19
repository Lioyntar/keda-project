import hashlib
import json
import os
import signal
import socket
import tempfile
import time

import redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# --- Ρυθμίσεις από environment variables (ίδιο image, διαφορετικές ρυθμίσεις) ---
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
QUEUE = os.getenv("QUEUE_NAME", "jobs")
DONE_LIST = os.getenv("DONE_LIST", "done")          # v2: εδώ γράφεται κάθε ολοκληρωμένο job
DEFAULT_ITERATIONS = int(os.getenv("WORK_ITERATIONS", "300000"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
HEARTBEAT_FILE = os.getenv("HEARTBEAT_FILE", os.path.join(tempfile.gettempdir(), "heartbeat"))
POD = socket.gethostname()

# --- Μετρικές για το Prometheus ---
JOBS_DONE = Counter("worker_jobs_processed_total", "Jobs processed")
JOB_TIME = Histogram("worker_job_processing_seconds", "Processing time per job")
WAIT_TIME = Histogram(
    "worker_job_queue_wait_seconds", "Time a job waited in the queue",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
BUSY = Gauge("worker_busy", "1 while processing a job, 0 when idle")

# --- Graceful shutdown: στο scale-in το Kubernetes στέλνει SIGTERM ---
running = True


def stop(signum, frame):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def heartbeat():
    """Γράφει την τρέχουσα ώρα - το liveness probe ελέγχει ότι είναι πρόσφατη."""
    with open(HEARTBEAT_FILE, "w") as f:
        f.write(str(time.time()))


def process(job):
    """Πραγματική δουλειά: παραγωγή κλειδιού με PBKDF2-SHA256."""
    salt = str(job["id"]).encode()
    iterations = job.get("iterations", DEFAULT_ITERATIONS)
    hashlib.pbkdf2_hmac("sha256", b"subscriber-secret", salt, iterations)


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    start_http_server(METRICS_PORT)
    print(f"[{POD}] waiting for jobs on '{QUEUE}' at {REDIS_HOST}:{REDIS_PORT}", flush=True)

    while running:
        heartbeat()
        item = r.blpop(QUEUE, timeout=1)   # περιμένει έως 1s για job
        if item is None:
            continue                       # άδεια ουρά -> ξαναέλεγξε το running
        job = json.loads(item[1])

        started = time.time()
        wait = started - job["enqueued_at"]
        WAIT_TIME.observe(wait)
        BUSY.set(1)
        with JOB_TIME.time():
            process(job)
        BUSY.set(0)
        finished = time.time()
        JOBS_DONE.inc()

        # v2: καταγραφή κάθε job για την αξιολόγηση (ερώτημα 6)
        r.rpush(DONE_LIST, json.dumps({
            "id": job["id"], "enq": job["enqueued_at"],
            "start": round(started, 4), "end": round(finished, 4), "pod": POD,
        }))

    print(f"[{POD}] shutting down gracefully", flush=True)


if __name__ == "__main__":
    main()
