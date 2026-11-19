"""
Ανάλυση πειραμάτων (ερώτημα 6).

    python scripts/analyze.py results/keda-combined_2026...            # ένα πείραμα
    python scripts/analyze.py results/keda-queue_... results/keda-combined_... results/hpa-cpu_...   # σύγκριση

Για κάθε πείραμα φτιάχνει <run>/timeline.png: έξι γραφήματα στον ίδιο άξονα χρόνου
(φορτίο, ουρά, σήμα του autoscaler, replicas, καθυστέρηση, CPU).
Με πολλά πειράματα φτιάχνει και results/comparison.png και results/summary.csv / summary.md.
"""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

PHASE_COLORS = {"low": "#e8f1f2", "spike": "#fde2c8", "high": "#fbd3d3", "ramp_down": "#efe6f7", "idle": "#f2f2f2"}
RUN_COLORS = {"keda-combined": "#2E8B8B", "keda-queue": "#F4A23B", "hpa-cpu": "#D1495B"}
LABELS = {"keda-combined": "KEDA (ζήτηση: throughput + ουρά)", "keda-queue": "KEDA (μόνο μήκος ουράς)",
          "hpa-cpu": "HPA (CPU)"}
SIGNAL_UNITS = {"keda-combined": "ζήτηση ανά replica (jobs/s)", "keda-queue": "jobs στην ουρά ανά replica",
                "hpa-cpu": "CPU ανά replica (% του request)"}
BIN = 10  # δευτερόλεπτα ανά κάδο για ρυθμούς και καθυστέρηση


def load_run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    load = pd.read_csv(os.path.join(run_dir, "load.csv"))
    samples = pd.read_csv(os.path.join(run_dir, "samples.csv"))
    jobs = pd.read_csv(os.path.join(run_dir, "jobs.csv"))
    t0 = float(load["unix_time"].iloc[0])
    samples["t"] = samples["ts"] - t0
    for c in ("replicas", "ready"):
        samples[c] = samples[c].ffill().fillna(0)
    # Το Kubernetes γράφει -9.2e15 όταν δεν μπορεί να διαβάσει τη μετρική: αυτές οι τιμές είναι άκυρες
    samples["metric_invalid"] = samples["hpa_metric"].abs() > 1e9
    samples.loc[samples["metric_invalid"], "hpa_metric"] = np.nan
    jobs["t_enq"] = jobs["enq"] - t0
    jobs["t_end"] = jobs["end"] - t0
    jobs["wait"] = (jobs["start"] - jobs["enq"]).clip(lower=0)
    jobs["service"] = jobs["end"] - jobs["start"]
    phases = load.groupby("phase", sort=False)["t"].agg(["min", "max"])
    tick = float(load["t"].diff().median()) if len(load) > 1 else 0.25
    phases["max"] = phases["max"] + tick
    return dict(dir=run_dir, meta=meta, mode=meta["mode"], load=load, samples=samples, jobs=jobs,
                phases=phases, tick=tick, end=float(load["t"].iloc[-1] + tick))


def mu_hat(run):
    """Ρυθμός εξυπηρέτησης ενός worker = 1 / διάμεσος χρόνος επεξεργασίας."""
    s = run["jobs"]["service"]
    return 1.0 / s.median() if len(s) else float("nan")


def needed_series(run, mu, maxr=10):
    load = run["load"]
    need = np.ceil(load["target_rate"] / mu).clip(upper=maxr)
    return load["t"], need


def binned(run):
    load, jobs = run["load"], run["jobs"]
    end = max(run["end"], float(jobs["t_end"].max()) if len(jobs) else 0) + BIN
    edges = np.arange(0, end + BIN, BIN)
    arr = np.histogram(load["t"], bins=edges, weights=load["sent"])[0] / BIN
    done = np.histogram(jobs["t_end"], bins=edges)[0] / BIN
    idx = np.digitize(jobs["t_enq"], edges) - 1
    p50, p95 = np.full(len(edges) - 1, np.nan), np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        w = jobs["wait"].values[idx == b]
        if len(w):
            p50[b], p95[b] = np.percentile(w, 50), np.percentile(w, 95)
    return pd.DataFrame(dict(t=edges[:-1], arrivals=arr, completions=done, p50=p50, p95=p95))


def shade(ax, run, label=False):
    for name, row in run["phases"].iterrows():
        ax.axvspan(row["min"], row["max"], color=PHASE_COLORS.get(name, "#f5f5f5"), zorder=0)
        if label:
            ax.text((row["min"] + row["max"]) / 2, 0.96, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, color="#555")


def summary(run):
    jobs, s, ph, mode = run["jobs"], run["samples"], run["phases"], run["mode"]
    mu = mu_hat(run)
    out = dict(run=os.path.basename(run["dir"].rstrip("/\\")), mode=mode)
    out["jobs_sent"] = int(run["load"]["sent"].sum())
    out["jobs_done"] = len(jobs)
    out["mu_jobs_per_s"] = round(mu, 2)
    w = jobs["wait"]
    for q in (50, 95, 99):
        out[f"wait_p{q}_s"] = round(float(np.percentile(w, q)), 2) if len(w) else None
    out["wait_max_s"] = round(float(w.max()), 2) if len(w) else None
    out["jobs_wait_over_1s_pct"] = round(100 * float((w > 1).mean()), 1) if len(w) else None
    out["jobs_wait_over_5s_pct"] = round(100 * float((w > 5).mean()), 1) if len(w) else None
    out["max_queue"] = int(s["queue_len"].max())

    # Αντίδραση στο spike
    if "spike" in ph.index:
        t_sp = ph.loc["spike", "min"]
        before = s[s["t"] <= t_sp]
        r0 = int(before["replicas"].iloc[-1]) if len(before) else int(s["replicas"].iloc[0])
        after = s[s["t"] >= t_sp]
        up = after[after["replicas"] > r0]
        out["spike_first_scaleup_s"] = round(float(up["t"].iloc[0] - t_sp), 1) if len(up) else None
        rate_sp = float(run["load"].loc[run["load"]["phase"] == "spike", "target_rate"].max())
        need = int(min(10, math.ceil(rate_sp / mu)))
        out["spike_replicas_needed"] = need
        ok = after[after["ready"] >= need]
        out["spike_time_to_needed_ready_s"] = round(float(ok["t"].iloc[0] - t_sp), 1) if len(ok) else None
        peak_t = float(after.loc[after["queue_len"].idxmax(), "t"]) if len(after) else t_sp
        cleared = after[(after["t"] > peak_t) & (after["queue_len"] < 20)]
        out["spike_backlog_cleared_s"] = round(float(cleared["t"].iloc[0] - t_sp), 1) if len(cleared) else None

    # Κόστος: replica-seconds και CPU μέσα στη διάρκεια του σεναρίου
    win = s[(s["t"] >= 0) & (s["t"] <= run["end"])]
    out["replica_seconds"] = int(np.trapezoid(win["replicas"], win["t"])) if len(win) > 1 else None
    out["cpu_core_seconds"] = int(np.trapezoid(win["cpu_cores"].fillna(0), win["t"])) if len(win) > 1 else None
    busy = float(jobs["service"].sum())
    out["worker_utilization_pct"] = round(100 * busy / out["replica_seconds"], 1) if out["replica_seconds"] else None
    out["scaling_changes"] = int((s["replicas"].diff().fillna(0) != 0).sum())
    out["metric_unavailable_samples"] = int(s["metric_invalid"].sum())

    # Αποκλιμάκωση: από την αρχή της ηρεμίας μέχρι το ελάχιστο
    if "idle" in ph.index:
        t_idle = ph.loc["idle", "min"]
        after = s[s["t"] >= t_idle]
        if len(after):
            rmin = int(after["replicas"].min())
            hit = after[after["replicas"] == rmin]
            out["idle_min_replicas"] = rmin
            out["idle_time_to_min_s"] = round(float(hit["t"].iloc[0] - t_idle), 1)
    # Μέσα replicas ανά φάση σε σχέση με όσα χρειάζονται
    for name, row in ph.iterrows():
        seg = s[(s["t"] >= row["min"]) & (s["t"] < row["max"])]
        if len(seg):
            out[f"avg_replicas_{name}"] = round(float(seg["replicas"].mean()), 1)
    return out


def plot_run(run):
    mode, s, load = run["mode"], run["samples"], run["load"]
    b = binned(run)
    mu = mu_hat(run)
    color = RUN_COLORS.get(mode, "#333")
    fig, ax = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(f"{LABELS.get(mode, mode)}  ·  {os.path.basename(run['dir'].rstrip('/'))}", fontsize=14, y=0.995)
    for i, a in enumerate(ax):
        shade(a, run, label=(i == 0))
        a.grid(alpha=0.3)

    a = ax[0]
    a.step(load["t"], load["target_rate"], where="post", color="#333", lw=1.5, label="ρυθμός στόχος")
    a.plot(b["t"] + BIN / 2, b["arrivals"], color="#888", lw=1, label=f"αφίξεις (μέσος {BIN}s)")
    a.plot(b["t"] + BIN / 2, b["completions"], color=color, lw=1.8, label="ολοκληρώσεις")
    a.set_ylabel("jobs / s")
    a.set_title("1. Ένταση φορτίου", loc="left", fontsize=11)
    a.set_ylim(top=a.get_ylim()[1] * 1.15)
    a.legend(loc="upper right", fontsize=9)

    a = ax[1]
    a.plot(s["t"], s["queue_len"], color="#555", lw=1.5)
    a.set_ylabel("jobs")
    a.set_title("2. Μήκος ουράς (jobs σε αναμονή)", loc="left", fontsize=11)

    a = ax[2]
    a.plot(s["t"], s["hpa_metric"], color=color, lw=1.5, label="σήμα (όπως το βλέπει ο autoscaler)")
    if s["hpa_target"].notna().any():
        a.axhline(float(s["hpa_target"].dropna().iloc[0]), color="#333", ls="--", lw=1, label="στόχος")
    bad = s[s["metric_invalid"]]
    if len(bad):
        a.scatter(bad["t"], np.zeros(len(bad)), marker="x", color="#D1495B", s=18, zorder=5,
                  label="μετρική μη διαθέσιμη")
    a.set_ylabel(SIGNAL_UNITS.get(mode, "σήμα"))
    a.set_title("3. Σήμα κλιμάκωσης και στόχος", loc="left", fontsize=11)
    a.legend(loc="upper right", fontsize=9)

    a = ax[3]
    tn, need = needed_series(run, mu)
    a.step(tn, need, where="post", color="#333", ls="--", lw=1.2, label=f"χρειάζονται (λ/μ, μ≈{mu:.1f})")
    if s["hpa_desired"].notna().any():
        a.step(s["t"], s["hpa_desired"], where="post", color="#aaa", lw=1, label="απόφαση autoscaler")
    a.step(s["t"], s["replicas"], where="post", color=color, lw=2, label="replicas")
    a.step(s["t"], s["ready"], where="post", color=color, lw=1, ls=":", label="έτοιμα")
    a.set_ylabel("workers")
    a.set_ylim(-0.3, 11)
    a.set_title("4. Αριθμός replicas", loc="left", fontsize=11)
    a.legend(loc="upper right", fontsize=9)

    a = ax[4]
    a.plot(b["t"] + BIN / 2, b["p50"], color=color, lw=1.5, label="p50")
    a.plot(b["t"] + BIN / 2, b["p95"], color=color, lw=1, ls="--", label="p95")
    a.set_ylabel("δευτερόλεπτα")
    a.set_title("5. Χρόνος αναμονής στην ουρά (ανά ώρα άφιξης)", loc="left", fontsize=11)
    a.legend(loc="upper right", fontsize=9)

    a = ax[5]
    a.plot(s["t"], s["cpu_cores"], color=color, lw=1.5, label="CPU σε χρήση")
    a.step(s["t"], s["replicas"] * 0.5, where="post", color="#999", lw=1, ls="--", label="CPU δεσμευμένη (requests)")
    a.set_ylabel("πυρήνες")
    a.set_title("6. Πόροι", loc="left", fontsize=11)
    a.set_xlabel("χρόνος από την αρχή του σεναρίου (s)")
    a.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = os.path.join(run["dir"], "timeline.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_comparison(runs, out):
    fig, ax = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    ref = runs[0]
    for a in ax:
        shade(a, ref)
        a.grid(alpha=0.3)
    mu = np.nanmedian([mu_hat(r) for r in runs])
    tn, need = needed_series(ref, mu)
    ax[0].step(tn, need, where="post", color="#333", ls="--", lw=1.2, label="χρειάζονται (λ/μ)")
    for r in runs:
        c = RUN_COLORS.get(r["mode"], None)
        lab = LABELS.get(r["mode"], r["mode"])
        ax[0].step(r["samples"]["t"], r["samples"]["replicas"], where="post", color=c, lw=2, label=lab)
        ax[1].plot(r["samples"]["t"], r["samples"]["queue_len"], color=c, lw=1.5, label=lab)
        b = binned(r)
        ax[2].plot(b["t"] + BIN / 2, b["p95"], color=c, lw=1.5, label=lab)
    ax[0].set_ylabel("replicas"); ax[0].set_title("Replicas", loc="left")
    ax[1].set_ylabel("jobs"); ax[1].set_title("Μήκος ουράς", loc="left")
    ax[2].set_ylabel("s"); ax[2].set_title(f"p95 χρόνου αναμονής (κάδοι {BIN}s)", loc="left")
    ax[2].set_xlabel("χρόνος (s)")
    for a in ax:
        a.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def to_markdown(df):
    cols = [str(c) for c in df.columns]
    lines = ["| μετρική | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for idx, row in df.iterrows():
        lines.append(f"| {idx} | " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main():
    dirs = [d for d in sys.argv[1:] if os.path.isdir(d)]
    if not dirs:
        sys.exit(__doc__)
    runs = [load_run(d) for d in dirs]
    rows = []
    for r in runs:
        print(f"{r['dir']}: timeline -> {plot_run(r)}")
        rows.append(summary(r))
    df = pd.DataFrame(rows).set_index("run").T
    print("\n" + df.to_string())
    if len(runs) > 1:
        os.makedirs("results", exist_ok=True)
        plot_comparison(runs, "results/comparison.png")
        df.to_csv("results/summary.csv")
        with open("results/summary.md", "w", encoding="utf-8") as f:
            f.write(to_markdown(df))
        print("\nComparison -> results/comparison.png, results/summary.csv, results/summary.md")
    else:
        df.to_csv(os.path.join(runs[0]["dir"], "summary.csv"))


if __name__ == "__main__":
    main()
