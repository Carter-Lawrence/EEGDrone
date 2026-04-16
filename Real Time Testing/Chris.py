"""
Tennis Match Analytics Engine — Full Reference Implementation
=============================================================
Track match data, calculate performance stats, identify clutch moments,
and visualize trends with matplotlib.
"""

import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────────────────────
# 1.  DATA LAYER — store everything in plain dictionaries
# ──────────────────────────────────────────────────────────────

def create_match(opponent, surface, date, tournament="Practice"):
    """Return a new match dictionary pre-loaded with empty stat buckets."""
    return {
        "opponent": opponent,
        "surface": surface,          # "hard", "clay", "grass"
        "date": date,                # "YYYY-MM-DD"
        "tournament": tournament,
        "sets_won": 0,
        "sets_lost": 0,
        "result": None,              # "W" or "L" — set after match ends

        # Serve stats
        "first_serves_in": 0,
        "first_serve_attempts": 0,
        "aces": 0,
        "double_faults": 0,
        "service_points_won": 0,
        "service_points_played": 0,

        # Return stats
        "return_points_won": 0,
        "return_points_played": 0,

        # Clutch / pressure stats
        "break_points_saved": 0,
        "break_points_faced": 0,
        "break_points_converted": 0,
        "break_points_opportunities": 0,
        "tiebreaks_won": 0,
        "tiebreaks_played": 0,
        "deuce_points_won": 0,
        "deuce_points_played": 0,

        # Points totals
        "total_points_won": 0,
        "total_points_played": 0,
    }


def record_result(match, sets_won, sets_lost):
    """Finalize a match with sets won/lost and derive W/L."""
    match["sets_won"] = sets_won
    match["sets_lost"] = sets_lost
    match["result"] = "W" if sets_won > sets_lost else "L"


# ──────────────────────────────────────────────────────────────
# 2.  CALCULATION HELPERS — pure functions, easy to test
# ──────────────────────────────────────────────────────────────

def safe_pct(numerator, denominator):
    """Return a percentage (0-100) or 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 1)


def serve_stats(match):
    """Calculate a dictionary of serve-related percentages."""
    return {
        "first_serve_pct": safe_pct(match["first_serves_in"],
                                    match["first_serve_attempts"]),
        "ace_rate": safe_pct(match["aces"],
                             match["service_points_played"]),
        "double_fault_rate": safe_pct(match["double_faults"],
                                      match["service_points_played"]),
        "service_points_won_pct": safe_pct(match["service_points_won"],
                                           match["service_points_played"]),
    }


def return_stats(match):
    """Calculate return-game percentages."""
    return {
        "return_points_won_pct": safe_pct(match["return_points_won"],
                                          match["return_points_played"]),
    }


def clutch_stats(match):
    """Calculate pressure-situation percentages."""
    return {
        "break_points_saved_pct": safe_pct(match["break_points_saved"],
                                           match["break_points_faced"]),
        "break_points_converted_pct": safe_pct(match["break_points_converted"],
                                                match["break_points_opportunities"]),
        "tiebreak_win_pct": safe_pct(match["tiebreaks_won"],
                                     match["tiebreaks_played"]),
        "deuce_win_pct": safe_pct(match["deuce_points_won"],
                                  match["deuce_points_played"]),
    }


def clutch_score(match):
    """
    Composite 'clutch rating' (0-100) that blends pressure stats.
    Weights: break-point saving 35%, break-point converting 30%,
             tiebreaks 20%, deuce points 15%.
    """
    c = clutch_stats(match)
    score = (c["break_points_saved_pct"]     * 0.35
           + c["break_points_converted_pct"] * 0.30
           + c["tiebreak_win_pct"]           * 0.20
           + c["deuce_win_pct"]              * 0.15)
    return round(score, 1)


def overall_points_won_pct(match):
    """Total points won percentage."""
    return safe_pct(match["total_points_won"], match["total_points_played"])


# ──────────────────────────────────────────────────────────────
# 3.  MULTI-MATCH AGGREGATION
# ──────────────────────────────────────────────────────────────

def win_loss_record(matches):
    """Return (wins, losses) tuple from a list of matches."""
    wins = sum(1 for m in matches if m["result"] == "W")
    losses = sum(1 for m in matches if m["result"] == "L")
    return wins, losses


def avg_stat(matches, numerator_key, denominator_key):
    """Average a percentage stat across multiple matches."""
    total_num = sum(m[numerator_key] for m in matches)
    total_den = sum(m[denominator_key] for m in matches)
    return safe_pct(total_num, total_den)


def surface_breakdown(matches):
    """Group win/loss records by surface."""
    surfaces = {}
    for m in matches:
        s = m["surface"]
        if s not in surfaces:
            surfaces[s] = {"W": 0, "L": 0}
        if m["result"]:
            surfaces[s][m["result"]] += 1
    return surfaces


def season_summary(matches):
    """Print a formatted season summary to the console."""
    w, l = win_loss_record(matches)
    print("=" * 50)
    print("           SEASON SUMMARY")
    print("=" * 50)
    print(f"  Record:  {w}W – {l}L  "
          f"({safe_pct(w, w + l)}% win rate)")
    print()

    avg_1st = avg_stat(matches, "first_serves_in", "first_serve_attempts")
    avg_svp = avg_stat(matches, "service_points_won", "service_points_played")
    avg_rtn = avg_stat(matches, "return_points_won", "return_points_played")
    avg_bps = avg_stat(matches, "break_points_saved", "break_points_faced")
    avg_bpc = avg_stat(matches, "break_points_converted",
                       "break_points_opportunities")

    print(f"  Avg 1st Serve %:        {avg_1st}%")
    print(f"  Avg Svc Pts Won %:      {avg_svp}%")
    print(f"  Avg Return Pts Won %:   {avg_rtn}%")
    print(f"  Avg BP Saved %:         {avg_bps}%")
    print(f"  Avg BP Converted %:     {avg_bpc}%")
    print()

    print("  Surface breakdown:")
    for surface, record in surface_breakdown(matches).items():
        print(f"    {surface:8s}  {record['W']}W – {record['L']}L")
    print("=" * 50)


# ──────────────────────────────────────────────────────────────
# 4.  MATCH REPORT  — single-match deep dive
# ──────────────────────────────────────────────────────────────

def match_report(match):
    """Print a detailed single-match report."""
    sv = serve_stats(match)
    rt = return_stats(match)
    cl = clutch_stats(match)

    result_emoji = "✅" if match["result"] == "W" else "❌"
    print()
    print("─" * 50)
    print(f"  {result_emoji}  vs {match['opponent']}  "
          f"({match['date']}, {match['surface']})")
    print(f"     {match['tournament']}  —  "
          f"Sets: {match['sets_won']}-{match['sets_lost']}")
    print("─" * 50)

    print(f"  1st Serve %:          {sv['first_serve_pct']}%")
    print(f"  Ace Rate:             {sv['ace_rate']}%")
    print(f"  Double Fault Rate:    {sv['double_fault_rate']}%")
    print(f"  Svc Pts Won:          {sv['service_points_won_pct']}%")
    print(f"  Return Pts Won:       {rt['return_points_won_pct']}%")
    print(f"  BP Saved:             {cl['break_points_saved_pct']}%")
    print(f"  BP Converted:         {cl['break_points_converted_pct']}%")
    print(f"  Tiebreak Win %:       {cl['tiebreak_win_pct']}%")
    print(f"  Deuce Win %:          {cl['deuce_win_pct']}%")
    print(f"  Clutch Score:         {clutch_score(match)}")
    print(f"  Total Pts Won:        {overall_points_won_pct(match)}%")
    print("─" * 50)


# ──────────────────────────────────────────────────────────────
# 5.  VISUALIZATIONS
# ──────────────────────────────────────────────────────────────

def plot_serve_dashboard(match, save_path=None):
    """Bar chart of key serve metrics for one match."""
    sv = serve_stats(match)
    labels = ["1st Serve %", "Ace Rate", "Dbl Fault Rate", "Svc Pts Won %"]
    values = [sv["first_serve_pct"], sv["ace_rate"],
              sv["double_fault_rate"], sv["service_points_won_pct"]]
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val}%", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylim(0, max(values) * 1.25 if values else 100)
    ax.set_ylabel("Percentage")
    ax.set_title(f"Serve Dashboard — vs {match['opponent']}  "
                 f"({match['date']})", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Chart saved to {save_path}")
    plt.show()


def plot_clutch_radar(match, save_path=None):
    """Simple horizontal bar chart of clutch metrics."""
    cl = clutch_stats(match)
    labels = ["BP Saved %", "BP Converted %", "Tiebreak Win %", "Deuce Win %"]
    values = [cl["break_points_saved_pct"], cl["break_points_converted_pct"],
              cl["tiebreak_win_pct"], cl["deuce_win_pct"]]
    colors = ["#f59e0b", "#ef4444", "#06b6d4", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Percentage")
    ax.set_title(f"Clutch Performance — vs {match['opponent']}",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Chart saved to {save_path}")
    plt.show()


def plot_season_trend(matches, stat_name, num_key, den_key, save_path=None):
    """Line chart showing a stat trending across matches."""
    dates = [m["date"] for m in matches]
    values = [safe_pct(m[num_key], m[den_key]) for m in matches]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates, values, marker="o", linewidth=2.5, color="#2563eb",
            markerfacecolor="white", markeredgewidth=2, markersize=8)
    for i, (d, v) in enumerate(zip(dates, values)):
        ax.annotate(f"{v}%", (d, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_ylabel("Percentage")
    ax.set_title(f"{stat_name} — Season Trend",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Chart saved to {save_path}")
    plt.show()


def plot_surface_win_rates(matches, save_path=None):
    """Grouped bar chart of wins vs losses by surface."""
    breakdown = surface_breakdown(matches)
    surfaces = list(breakdown.keys())
    wins = [breakdown[s]["W"] for s in surfaces]
    losses = [breakdown[s]["L"] for s in surfaces]

    x = range(len(surfaces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], wins, width,
           label="Wins", color="#16a34a", edgecolor="white")
    ax.bar([i + width / 2 for i in x], losses, width,
           label="Losses", color="#dc2626", edgecolor="white")

    ax.set_xticks(list(x))
    ax.set_xticklabels([s.title() for s in surfaces])
    ax.set_ylabel("Matches")
    ax.set_title("Win / Loss by Surface", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Chart saved to {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 6.  DEMO — sample data + run everything
# ──────────────────────────────────────────────────────────────

def build_demo_matches():
    """Return a list of sample matches with realistic-ish numbers."""
    m1 = create_match("Alex R.", "hard", "2026-02-15", "Winter Open L5")
    m1.update({
        "first_serves_in": 42, "first_serve_attempts": 68,
        "aces": 5, "double_faults": 3,
        "service_points_won": 48, "service_points_played": 72,
        "return_points_won": 30, "return_points_played": 65,
        "break_points_saved": 4, "break_points_faced": 6,
        "break_points_converted": 3, "break_points_opportunities": 5,
        "tiebreaks_won": 1, "tiebreaks_played": 1,
        "deuce_points_won": 7, "deuce_points_played": 11,
        "total_points_won": 78, "total_points_played": 137,
    })
    record_result(m1, 2, 1)

    m2 = create_match("Jordan T.", "clay", "2026-03-01", "Spring Clay Series")
    m2.update({
        "first_serves_in": 38, "first_serve_attempts": 65,
        "aces": 2, "double_faults": 5,
        "service_points_won": 40, "service_points_played": 70,
        "return_points_won": 35, "return_points_played": 68,
        "break_points_saved": 2, "break_points_faced": 5,
        "break_points_converted": 4, "break_points_opportunities": 7,
        "tiebreaks_won": 0, "tiebreaks_played": 0,
        "deuce_points_won": 5, "deuce_points_played": 9,
        "total_points_won": 75, "total_points_played": 138,
    })
    record_result(m2, 2, 0)

    m3 = create_match("Sam K.", "hard", "2026-03-15", "Level 5 Regional")
    m3.update({
        "first_serves_in": 35, "first_serve_attempts": 70,
        "aces": 3, "double_faults": 7,
        "service_points_won": 38, "service_points_played": 74,
        "return_points_won": 28, "return_points_played": 66,
        "break_points_saved": 1, "break_points_faced": 4,
        "break_points_converted": 2, "break_points_opportunities": 6,
        "tiebreaks_won": 0, "tiebreaks_played": 1,
        "deuce_points_won": 4, "deuce_points_played": 10,
        "total_points_won": 66, "total_points_played": 140,
    })
    record_result(m3, 1, 2)

    m4 = create_match("Riley M.", "hard", "2026-03-28", "Level 5 Regional")
    m4.update({
        "first_serves_in": 50, "first_serve_attempts": 72,
        "aces": 8, "double_faults": 2,
        "service_points_won": 55, "service_points_played": 75,
        "return_points_won": 32, "return_points_played": 62,
        "break_points_saved": 5, "break_points_faced": 5,
        "break_points_converted": 3, "break_points_opportunities": 4,
        "tiebreaks_won": 1, "tiebreaks_played": 1,
        "deuce_points_won": 8, "deuce_points_played": 10,
        "total_points_won": 87, "total_points_played": 137,
    })
    record_result(m4, 2, 0)

    return [m1, m2, m3, m4]


def main():
    matches = build_demo_matches()

    # Individual match reports
    for m in matches:
        match_report(m)

    # Season summary
    print()
    season_summary(matches)

    # ── Charts ──
    output_dir = "charts"
    os.makedirs(output_dir, exist_ok=True)

    # Serve dashboard for most recent match
    plot_serve_dashboard(matches[-1],
                         save_path=f"{output_dir}/serve_dashboard.png")

    # Clutch chart for the tough loss
    plot_clutch_radar(matches[2],
                      save_path=f"{output_dir}/clutch_radar.png")

    # Season trend: 1st-serve %
    plot_season_trend(matches, "1st Serve %",
                      "first_serves_in", "first_serve_attempts",
                      save_path=f"{output_dir}/first_serve_trend.png")

    # Surface win/loss
    plot_surface_win_rates(matches,
                           save_path=f"{output_dir}/surface_winloss.png")


if __name__ == "__main__":
    main()