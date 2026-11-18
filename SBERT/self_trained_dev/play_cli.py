"""Play the game in a terminal with the exported model (quick manual check before the web integration).

Usage:
    python play_cli.py --artifacts artifacts            # play
    python play_cli.py --artifacts artifacts --check    # verify files and run the golden cases only
"""
import argparse
import json
import time
from pathlib import Path

from lexical import GameSession, load_game, verify_artifacts

BACKEND = "self_trained"


def run_golden(game, folder: Path) -> bool:
    """Re-score golden_cases.json and report differences from the notebook."""
    golden = json.loads((folder / "golden_cases.json").read_text(encoding="utf-8"))
    bad, times = [], []
    for case in golden["cases"]:
        start = time.perf_counter()
        r = game.engine.evaluate_attack(case["guess"], case["target"], case["domain"], set(case["used"]),
                                        case["hp_max"])
        times.append((time.perf_counter() - start) * 1000)
        r = game.engine.resolve_damage(r, case["hp_max"], case["hp_now"])
        e = case["expected"]
        if r["valid"] != e["valid"] or (e["valid"] and (
                r["domain_match_type"] != e["domain_match_type"] or r["tier"] != e["tier"]
                or abs(r["actual_damage"] - e["actual_damage"]) > 1 or abs(r["similarity"] - e["similarity"]) > 1e-3)):
            bad.append((case["source"], case["target"], case["guess"]))
    times.sort()
    print(f"Golden cases: {len(golden['cases'])} | mismatches: {len(bad)} | "
          f"latency median {times[len(times) // 2]:.1f} ms, p95 {times[int(len(times) * 0.95)]:.1f} ms")
    for b in bad[:20]:
        print("  mismatch:", b)
    return not bad


def play(game, rounds: int, seed) -> None:
    session = GameSession.start(game.eligible, game.game_data.boss_names, rounds, seed)
    print("Type 1-3 English words per guess | 'hint' | 'skip' | 'quit'")
    while not session.finished:
        s = session.state()
        guess = input(f"[Stage {s['stage_number']}/{s['stage_count']} {s['domain']} · {s['enemy']} · "
                      f"target '{s['target'].upper()}' · HP {s['hp']}/{s['hp_max']}] > ").strip()
        if guess.lower() == "quit":
            session.end_game()
            break
        if guess.lower() == "hint":
            print("  Hint:", session.hint(game.engine))
            continue
        if guess.lower() == "skip":
            session.skip()
            continue
        event = session.attack(game.engine, guess)
        r = event["result"]
        if not event["counted"]:
            print("  Not counted:", r.get("reason", ""))
            continue
        print(f"  {r['word']}: gate {r['domain_match_type']} · r {r['relevance']:.2f} · "
              f"-{r['actual_damage']} HP · {r['tier']}")
        if event["stage_ended"]:
            print(f"  >>> {event['stage_ended']['enemy']} defeated in {event['stage_ended']['turns']} turns")
    for log in session.stage_logs:
        print(f"Stage {log['stage']} {log['enemy']} ({log['domain']}, '{log['target']}'): {log['status']}, "
              f"HP left {log['hp_left']}/{log['hp_max']}, {log['turns']} turns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default="artifacts", help="folder with the notebook export files")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="verify files and golden cases, then exit")
    args = parser.parse_args()
    folder = Path(args.artifacts)
    problems = verify_artifacts(folder, check_hashes=True)
    if problems:
        raise SystemExit(f"Artifacts problem: {problems}")
    start = time.perf_counter()
    game = load_game(BACKEND, folder)
    print(f"Loaded {BACKEND} ({game.manifest.get('model')}) in {time.perf_counter() - start:.1f} s | "
          f"eligible targets: { {d: len(t) for d, t in game.eligible.items()} }")
    if args.check:
        raise SystemExit(0 if run_golden(game, folder) else 1)
    play(game, args.rounds, args.seed)


if __name__ == "__main__":
    main()
