"""
Collector: κάθε λίγα δευτερόλεπτα γράφει σε CSV την κατάσταση του συστήματος.

  - queue_len           : μήκος ουράς (Redis LLEN)
  - replicas / ready    : πόσα worker pods ζητά το Deployment / πόσα είναι έτοιμα
  - cpu_cores           : πόση CPU καίνε συνολικά οι workers (metrics-server)
  - cpu_util_pct        : μέση CPU ανά worker ως % του request (αυτό βλέπει το HPA)
  - hpa_*               : τι «βλέπει» και τι αποφασίζει ο autoscaler (το HPA ή το HPA που φτιάχνει το KEDA)

Όλες οι χρονοσφραγίδες είναι από το ρολόι του Redis (ίδιο ρολόι με generator και workers).
Χρησιμοποιείται από το run_experiment.py, αλλά τρέχει και μόνο του:
    python scripts/collector.py results/test_samples.csv
"""
import csv
import sys
import threading
import time

import redis
from kubernetes import client, config

NAMESPACE = "default"
DEPLOYMENT = "worker"
LABEL = "app=worker"
CPU_REQUEST_CORES = 0.5

FIELDS = ["ts", "queue_len", "replicas", "ready", "pods_with_metrics", "cpu_cores", "cpu_util_pct",
          "hpa_name", "hpa_current", "hpa_desired", "hpa_metric", "hpa_target", "hpa_metric_name"]

_SUFFIX = {"n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
           "Ki": 1024.0, "Mi": 1024.0 ** 2, "Gi": 1024.0 ** 3, "Ti": 1024.0 ** 4}


def parse_quantity(q):
    """Kubernetes quantity -> float. '250m' -> 0.25, '1234567n' -> 0.0012, '6500m' -> 6.5, '3' -> 3.0"""
    if q is None:
        return None
    s = str(q).strip()
    for suf in ("Ki", "Mi", "Gi", "Ti", "n", "u", "m", "k", "M", "G", "T"):
        if s.endswith(suf):
            return float(s[: -len(suf)]) * _SUFFIX[suf]
    return float(s)


def redis_now(r):
    sec, usec = r.time()
    return sec + usec / 1e6


class Collector(threading.Thread):
    def __init__(self, out_csv, r, interval=2.0, queue="jobs"):
        super().__init__(daemon=True)
        self.out_csv, self.r, self.interval, self.queue = out_csv, r, interval, queue
        self._halt = threading.Event()
        self.errors = {}
        config.load_kube_config()
        self.apps = client.AppsV1Api()
        self.custom = client.CustomObjectsApi()
        self.hpa_api = client.AutoscalingV2Api()

    def stop(self):
        self._halt.set()

    def _err(self, where, e):
        if where not in self.errors:                    # κάθε είδος σφάλματος τυπώνεται μία φορά
            print(f"\n[collector] {where}: {type(e).__name__}: {e}", flush=True)
        self.errors[where] = self.errors.get(where, 0) + 1

    def sample(self):
        row = dict.fromkeys(FIELDS, "")
        try:
            row["ts"] = f"{redis_now(self.r):.3f}"
            row["queue_len"] = self.r.llen(self.queue)
        except Exception as e:
            self._err("redis", e)
            row["ts"] = f"{time.time():.3f}"

        try:
            d = self.apps.read_namespaced_deployment(DEPLOYMENT, NAMESPACE)
            row["replicas"] = d.spec.replicas or 0
            row["ready"] = d.status.ready_replicas or 0
        except Exception as e:
            self._err("deployment", e)

        try:
            pm = self.custom.list_namespaced_custom_object(
                "metrics.k8s.io", "v1beta1", NAMESPACE, "pods", label_selector=LABEL)
            per_pod = []
            for item in pm.get("items", []):
                cores = sum(parse_quantity(c["usage"]["cpu"]) for c in item.get("containers", []))
                per_pod.append(cores)
            row["pods_with_metrics"] = len(per_pod)
            row["cpu_cores"] = f"{sum(per_pod):.3f}"
            if per_pod:
                row["cpu_util_pct"] = f"{100 * sum(per_pod) / len(per_pod) / CPU_REQUEST_CORES:.1f}"
        except Exception as e:
            self._err("pod metrics", e)

        try:
            hpas = self.hpa_api.list_namespaced_horizontal_pod_autoscaler(NAMESPACE).items
            hpa = next((h for h in hpas if h.spec.scale_target_ref.name == DEPLOYMENT), None)
            if hpa is not None:
                row["hpa_name"] = hpa.metadata.name
                row["hpa_current"] = hpa.status.current_replicas
                row["hpa_desired"] = hpa.status.desired_replicas
                spec_m = (hpa.spec.metrics or [None])[0]
                if spec_m is not None and spec_m.type == "Resource":
                    row["hpa_target"] = spec_m.resource.target.average_utilization
                    row["hpa_metric_name"] = "cpu_utilization_pct"
                elif spec_m is not None and spec_m.type == "External":
                    t = spec_m.external.target
                    row["hpa_target"] = parse_quantity(t.average_value or t.value)
                    row["hpa_metric_name"] = spec_m.external.metric.name
                for m in hpa.status.current_metrics or []:
                    if m.type == "Resource" and m.resource is not None:
                        row["hpa_metric"] = m.resource.current.average_utilization
                    elif m.type == "External" and m.external is not None:
                        cur = m.external.current
                        row["hpa_metric"] = parse_quantity(cur.average_value or cur.value)
        except Exception as e:
            self._err("hpa", e)
        return row

    def run(self):
        with open(self.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            next_t = time.time()
            while not self._halt.is_set():
                w.writerow(self.sample())
                f.flush()
                next_t += self.interval
                self._halt.wait(max(0.0, next_t - time.time()))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/samples_test.csv"
    c = Collector(out, redis.Redis(host="localhost", port=6379))
    c.start()
    print(f"Collecting to {out} - Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        c.stop()
        c.join()
