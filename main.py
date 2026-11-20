#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model (per trophy, stdlib only, no network, deterministic):
  * Earners and players are modelled separately: rate = earners / players.
    Both are fitted in log space against x = ln(1 + hours since release)
    over the case's last 24 hours of reads, anchored at the fitted level of
    the last read, then extrapolated to the target.  Splitting earners from
    players keeps a refresh jump in players from bending the rate trend the
    way a direct rate fit does.
  * Cross-trophy shrinkage: each trophy's earners slope is shrunk toward its
    tier's mean slope (platinum shrinks toward the global mean, it being the
    only platinum), with weight n / (n + 8) on its own slope.
  * Damped growth: the raw E/P growth factor g is raised to 0.7, shrinking
    the extrapolation 30% toward no change.  A log-linear extrapolation
    overshoots when the true curve saturates (trophy shares cannot grow
    forever) and undershoots nothing when it does not; 0.7 is the robust
    middle across a saturating world and a log-linear world (both tested
    locally), and it beats both the house persistence baseline and the
    full linear extrapolation in each.
  * Refresh timing: the answer is the page as of the last refresh before the
    target, so the forecast is evaluated half a typical refresh interval
    before the target when refreshes are visible in the history.
  * Flat history (no player growth, no refresh seen): the page is stale and
    a refresh is overdue.  The median then expects one big refresh before
    the target: players multiply by 6 (the launch-night refresh, 638 -> 3868
    in the rules) while new players have earned each trophy at ~5% of the
    old cohort's rate, so the median rate is r * (1 + 0.05 * 5) / 6.
  * Spread: logit sd from the fit residuals plus the refresh-step volatility
    observed in the history plus a 0.30 * sqrt(horizon in days) random walk,
    asymmetric (wider below the median: a refresh can only cut rates fast).
    The 0.30 random-walk coefficient is deliberately below the house
    baseline's 0.5: the median already carries the trend, so the residual
    uncertainty is smaller than a persistence forecast's, and a stacked
    spread was tested locally and loses to this narrower, data-driven one
    in both a saturating and a log-linear world.
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]
# Damping on the extrapolated growth factor (see docstring).
DAMP = 0.7
# Random-walk spread coefficient (logit sd per sqrt(day)).
# Flat-history fallback: expected refresh multiplier and the share of the
# old cohort's rate at which new players hold each trophy.
FLAT_JUMP = 6.0
FLAT_NEW_SHARE = 0.05


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(x):
    p = clamp_rate(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return (t - RELEASE).total_seconds() / 3600.0


def linfit(xs, ys):
    """Least-squares line y = a + b*x; returns (a, b, residual sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(ss / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, n


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0
    t_target = hours_since_release(target)
    x_target = math.log1p(max(0.0, t_target))

    # Build read series: (t_hours, x, players, [rates]).
    reads = []
    for h in history:
        t = hours_since_release(parse_time(h["slot"]))
        reads.append((t, math.log1p(max(0.0, t)), h["players_tracked"],
                      h["rates"]))
    reads.sort()
    if not reads:
        # No history at all: fall back to a flat 1% median with wide spread.
        return [{"id": tr["id"], "q": row_quantiles(1.0, 1.0)}
                for tr in trophies]

    # Recent window: last 24 hours of reads (or all of them).
    last_t = reads[-1][0]
    window = [r for r in reads if last_t - r[0] <= 24.0] or reads
    x_last = window[-1][1]

    # Players fit over the window.
    xs_p = [r[1] for r in window]
    ys_p = [math.log(max(1.0, r[2])) for r in window]
    a_p, b_p, sd_p, _ = linfit(xs_p, ys_p)
    p_last_fit = math.exp(a_p + b_p * x_last)
    p_first = reads[0][2]
    p_grew = p_last_fit > 1.05 * max(1.0, p_first) or b_p > 0.01

    # Refresh events in the full history: player jumps over 5% between reads.
    refresh_gaps = []
    refresh_drops = []  # logit rate changes at refreshes, per trophy
    prev = None
    for r in reads:
        if prev is not None and r[2] > 1.05 * prev[2]:
            refresh_gaps.append(r[0] - prev[0])
            for tr, r_now, r_prev in zip(trophies, r[3], prev[3]):
                if r_now is not None and r_prev is not None and r_prev > 0:
                    refresh_drops.append(logit(r_now) - logit(r_prev))
        prev = r
    if refresh_gaps:
        refresh_interval = median(refresh_gaps)
    else:
        refresh_interval = 6.0  # hours; prior when none seen
    if refresh_drops:
        sd_refresh = math.sqrt(sum(d * d for d in refresh_drops) /
                               len(refresh_drops))
        sd_refresh = min(1.0, max(0.15, sd_refresh))
    else:
        sd_refresh = 0.5 if not p_grew else 0.3

    # Per-trophy earners fits over the window.
    slopes = {}
    fitted_last_loge = {}
    resid = {}
    npts = {}
    for idx, tr in enumerate(trophies):
        xs, ys = [], []
        for r in window:
            rate = r[3][idx]
            if rate is None or rate <= 0:
                continue
            earners = rate / 100.0 * r[2]
            if earners < 0.5:
                continue
            xs.append(r[1])
            ys.append(math.log(earners))
        a, b, sd, n = linfit(xs, ys)
        slopes[tr["id"]] = b
        fitted_last_loge[tr["id"]] = a + b * x_last if n else None
        resid[tr["id"]] = sd
        npts[tr["id"]] = n

    # Tier-mean slopes (platinum has one trophy: use the global mean).
    tier_slopes = {}
    all_slopes = [b for b in slopes.values()]
    global_mean = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    for tier in ("bronze", "silver", "gold", "platinum"):
        vals = [slopes[tr["id"]] for tr in trophies if tr["tier"] == tier]
        tier_slopes[tier] = (sum(vals) / len(vals) if len(vals) >= 3
                             else global_mean)

    rows = []
    for idx, tr in enumerate(trophies):
        tid = tr["id"]
        tier = tr["tier"]
        last_rate = reads[-1][3][idx]
        if last_rate is None:
            last_rate = 0.5
        if last_rate >= 99.99:
            # A trophy everyone holds stays held.
            rows.append({"id": tid, "q": row_quantiles(99.99, 0.05)})
            continue

        if not p_grew:
            # Flat history: stale page, a refresh is overdue.
            med = clamp_rate(last_rate * (1.0 + FLAT_NEW_SHARE *
                                          (FLAT_JUMP - 1.0)) / FLAT_JUMP)
            s_lo, s_hi = 0.625, 0.4
            rows.append({"id": tid, "q": row_quantiles(med, (s_lo, s_hi))})
            continue

        # Trend regime: shrink the earners slope toward the tier mean.
        n = npts[tid]
        w = n / (n + 8.0) if n else 0.0
        b_shrunk = w * slopes[tid] + (1.0 - w) * tier_slopes[tier]
        # Anchor at the fitted earners level of the last read.
        if fitted_last_loge[tid] is None:
            log_e_last = math.log(max(0.5, last_rate / 100.0 *
                                      reads[-1][2]))
        else:
            log_e_last = fitted_last_loge[tid]
        # Evaluate half a refresh interval before the target: the answer is
        # the page as of the last refresh, not the trend at the target.
        t_eval = max(last_t, t_target - 0.5 * refresh_interval)
        x_eval = math.log1p(max(0.0, t_eval))
        log_e_t = log_e_last + b_shrunk * (x_eval - x_last)
        log_p_t = math.log(max(1.0, p_last_fit)) + b_p * (x_eval - x_last)
        raw_rate = 100.0 * math.exp(log_e_t - log_p_t)
        # Damp the raw growth factor toward no change (see docstring).
        g = raw_rate / max(0.01, last_rate)
        med = clamp_rate(last_rate * (g ** DAMP)) if g > 0 else raw_rate
        # Spread: fit residuals + refresh volatility + random walk.
        s_fit = min(1.0, max(0.15, resid[tid])) if n >= 3 else 0.4
        s_rw = 0.30 * math.sqrt(max(0.0, horizon_days))
        s_tot = math.sqrt(s_fit * s_fit + sd_refresh * sd_refresh +
                          s_rw * s_rw)
        s_tot = min(1.2, max(0.3, s_tot))
        rows.append({"id": tid,
                     "q": row_quantiles(med, (s_tot * 1.25, s_tot * 0.8))})

    rows.sort(key=lambda r: r["id"])
    return rows


def row_quantiles(med, s):
    """Five quantiles around median `med` in logit space.
    s is a logit sd, or (sd_below, sd_above) for an asymmetric spread."""
    if isinstance(s, tuple):
        s_lo, s_hi = s
    else:
        s_lo = s_hi = s
    lm = logit(med)
    qs = []
    for z in Z:
        sd = s_lo if z < 0 else s_hi
        qs.append(round(clamp_rate(inv_logit(lm + z * sd)), 4))
    # Enforce non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        rows = forecast_case(case)
        out = {"trophies": rows}
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
