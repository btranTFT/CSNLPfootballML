from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    n = 2400

    league = rng.choice(["EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1"], size=n)
    home_form = rng.normal(0.0, 1.0, size=n)
    away_form = rng.normal(0.0, 1.0, size=n)
    home_attack = rng.normal(0.0, 1.0, size=n)
    away_attack = rng.normal(0.0, 1.0, size=n)
    home_defense = rng.normal(0.0, 1.0, size=n)
    away_defense = rng.normal(0.0, 1.0, size=n)
    weather = rng.choice(["clear", "rain", "snow"], size=n, p=[0.72, 0.24, 0.04])
    derby = rng.integers(0, 2, size=n)

    linear_score = (
        0.7 * home_form
        - 0.65 * away_form
        + 0.45 * home_attack
        - 0.35 * away_attack
        + 0.2 * home_defense
        - 0.2 * away_defense
        + 0.25 * derby
    )

    draw_band = np.abs(linear_score) < 0.35
    target = np.where(
        draw_band,
        "draw",
        np.where(linear_score > 0, "home_win", "away_win"),
    )
    # Create mild class imbalance closer to real football outcomes.
    flip_idx = rng.choice(np.arange(n), size=int(0.08 * n), replace=False)
    target[flip_idx] = rng.choice(["home_win", "away_win"], size=len(flip_idx), p=[0.6, 0.4])

    start_date = np.datetime64("2021-08-01")
    match_date = start_date + rng.integers(0, 1000, size=n).astype("timedelta64[D]")

    df = pd.DataFrame(
        {
            "match_date": match_date.astype(str),
            "league": league,
            "weather": weather,
            "is_derby": derby,
            "home_form": home_form,
            "away_form": away_form,
            "home_attack": home_attack,
            "away_attack": away_attack,
            "home_defense": home_defense,
            "away_defense": away_defense,
            "target": target,
        }
    )
    out = Path("data/football_matches.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote sample data to: {out.resolve()}")


if __name__ == "__main__":
    main()
