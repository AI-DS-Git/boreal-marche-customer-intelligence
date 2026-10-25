"""Score the assistant on the 40-question golden set with ANY configured provider.

CI runs this offline for the rule provider (tests/test_assistant_golden.py). This script is how you
measure the real back-ends, which need a key or a running Ollama:

    # hosted
    set OPENAI_API_KEY=sk-...            (PowerShell: $env:OPENAI_API_KEY="sk-...")
    set EMBEDDING_PROVIDER=openai
    python scripts/eval_assistant.py --provider openai
    # local
    set OLLAMA_BASE_URL=http://localhost:11434
    set EMBEDDING_PROVIDER=ollama
    python scripts/eval_assistant.py --provider ollama

    # choose the similarity floor for the active embedder (do this before trusting the numbers)
    python scripts/eval_assistant.py --calibrate

Run from the repository root. Writes docs/head_c_eval_<provider>.json with --save.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.services.assistant_factory import build_assistant  # noqa: E402
from app.services.rules_service import RulesService  # noqa: E402
from tests.golden_eval import evaluate, load_golden  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="rule", choices=["rule", "openai", "ollama"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    s = get_settings()
    rules = RulesService(s.rules_path, s.popularity_path, s.products_path)
    svc = build_assistant(s, rules)
    print(f"retrieval={s.embedding_provider}  providers={sorted(svc.providers)}")

    if args.calibrate:
        index = svc.providers["rule"].policies
        good, bad = [], []
        for g in load_golden():
            top = index.search(g["question"], 1)
            if not top:
                continue
            if g["expected_policy"] and top[0]["policy"]["id"] == g["expected_policy"]:
                good.append(top[0]["score"])
            elif g["expected_refusal"] == "no_evidence":
                bad.append(top[0]["score"])
        print(f"answerable questions, correct top hit : min {min(good):.3f}  median {sorted(good)[len(good)//2]:.3f}")
        print(f"no-evidence questions, top score      : max {max(bad):.3f}")
        if min(good) > max(bad):
            print(f"clean separation -> suggested EMBEDDING_FLOOR={(min(good) + max(bad)) / 2:.2f}")
        else:
            print("scores overlap: pick the floor that favours refusing (a wrong refusal hands off to staff; "
                  "a wrong answer is worse). Current floor:", s.floor_for(s.embedding_provider))
        return

    if args.provider not in svc.providers:
        sys.exit(f"provider '{args.provider}' is not configured (set its key / base URL)")
    report = evaluate(svc, args.provider)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in report.items()
                      if k != "failures"}, indent=2))
    for f in report["failures"]:
        print(f"MISS {f['id']}: {f['question']!r} tool={f['got_tool']} (want {f['expected_tool']}) "
              f"refusal={f['got_refusal']} (want {f['expected_refusal']}) cites={f['citations']}")
    if args.save:
        out = Path("docs") / f"head_c_eval_{args.provider}.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print("saved", out)


if __name__ == "__main__":
    main()
