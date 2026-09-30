"""Evaluate a TTS system (paper Table 1): mean UTMOS, corpus MER %, mean CMI / ΔCMI %.

Env: cosyvoicenew (pure python: stdlib + cmi_dpo.common; no torch). Runs on a compute node
(CPU srun is enough) because no python runs on the login node.

Set rows come from --set_dir/cands.tsv (utt cand wav ...) or, failing that, --set_dir/wav.scp
(cand = --cand_label, default 'gt', matching 11_/12_/13_ in --wav_scp mode). They are joined on
(utt, cand) with mer.tsv (edits, n_ref), utmos.tsv (utmos) and cmi.tsv (cmi). Ground-truth CMI
per utt is taken from the cand=='gt' rows of --cmi_gt_tsv (default: the same file as --cmi_tsv);
ΔCMI(utt,cand) = |cmi(utt,cand) - cmi(utt,'gt')|.  Corpus MER = sum(edits)/sum(n_ref) over rows
with n_ref > 0. All CMI values are stored in [0,1] and reported as %.
Duplicate (utt,cand) set rows are collapsed (last wins). For a synthesized corpus whose ids were
prefixed by 20_synthesize.py --utt_prefix, pass the same --utt_prefix so the gt CMI join works.
eval.json is strict JSON: undefined metrics (no rows) are written as null, printed as nan.
No configured paths are needed; --show_paths prints them (cmi_dpo.paths) and exits.

Example:
  python -u scripts/eval_tts.py --set_dir exp/round1 --mer_tsv exp/round1/mer.tsv \
      --utmos_tsv exp/round1/utmos.tsv --cmi_tsv exp/round1/cmi.tsv --cand 0 --set_name devman_sft
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cmi_dpo import common, paths  # noqa: E402

LOG = logging.getLogger("eval_tts")


class ShowPathsAction(argparse.Action):
    """--show_paths: print cmi_dpo.paths.describe() and exit, before the required flags are checked."""

    def __init__(self, option_strings: list[str], dest: str, **kwargs: Any) -> None:
        super().__init__(option_strings, dest, nargs=0, default=argparse.SUPPRESS,
                         help=kwargs.get("help", "print the configured paths (cmi_dpo.paths) and exit"))

    def __call__(self, parser: argparse.ArgumentParser, namespace: argparse.Namespace,
                 values: Any, option_string: str | None = None) -> None:
        print(paths.describe())
        parser.exit()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge critic TSVs for one TTS set and report Table-1 metrics (env cosyvoicenew, pure python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--set_dir", required=True, help="Dir holding cands.tsv (preferred) or wav.scp of the set.")
    p.add_argument("--mer_tsv", required=True, help="Output of 11_score_mer.py for this set.")
    p.add_argument("--utmos_tsv", required=True, help="Output of 12_score_utmos.py for this set.")
    p.add_argument("--cmi_tsv", required=True, help="Output of 13_score_cmi.py for this set (candidate rows).")
    p.add_argument("--cmi_gt_tsv", default=None,
                   help="TSV holding the cand=='gt' rows (real-speech CMI per utt); default = --cmi_tsv.")
    p.add_argument("--cand", default=None,
                   help="Only evaluate rows with this cand value (e.g. '0' = first candidate); default = all.")
    p.add_argument("--cand_label", default="gt", help="cand value assigned to wav.scp rows when no cands.tsv exists.")
    p.add_argument("--utt_prefix", default="",
                   help="Prefix that 20_synthesize.py --utt_prefix added to the set's utt ids (e.g. 'syn_'); it is "
                        "stripped before looking up the ground-truth CMI of the real utterance.")
    p.add_argument("--set_name", default=None, help="Name recorded in eval.json; default = basename of --set_dir.")
    p.add_argument("--out", default=None, help="Where to write eval.json; default = <set_dir>/eval.json.")
    p.add_argument("--show_paths", action=ShowPathsAction)
    return p


def dedupe_rows(rows: list[dict]) -> list[dict]:
    """Collapse duplicate (utt, cand) rows (append-only cands.tsv): last occurrence wins, first position kept."""
    uniq: dict[tuple[str, str], dict] = {}
    for r in rows:
        uniq[(r["utt"], r["cand"])] = r
    if len(uniq) != len(rows):
        LOG.warning("%d duplicate (utt,cand) rows dropped (last occurrence kept)", len(rows) - len(uniq))
    return list(uniq.values())


def load_set_rows(set_dir: str, cand_label: str) -> list[dict]:
    """Set rows {utt, cand, wav}: cands.tsv (deduped on (utt,cand)) or, failing that, wav.scp alone.

    wav.scp is parsed directly (no `text`/utt2dur needed: eval_tts uses neither transcript nor duration).
    """
    cands = os.path.join(set_dir, "cands.tsv")
    if os.path.exists(cands):
        rows = [{"utt": r["utt"], "cand": r["cand"], "wav": r["wav"]} for r in common.read_tsv(cands)]
        LOG.info("set rows from %s: %d", cands, len(rows))
        return dedupe_rows(rows)
    wav_scp = os.path.join(set_dir, "wav.scp")
    if not os.path.exists(wav_scp):
        raise SystemExit(f"neither cands.tsv nor wav.scp found in {set_dir}")
    wav = common._read_two_col(wav_scp)  # noqa: SLF001 - only wav paths are needed
    rows = [{"utt": u, "cand": cand_label, "wav": w} for u, w in wav.items()]
    LOG.info("set rows from %s: %d (cand=%s)", wav_scp, len(rows), cand_label)
    return rows


def gt_utt(utt: str, prefix: str) -> str:
    """Map a (possibly prefixed, see 20_synthesize.py --utt_prefix) set utt id to the real utt id."""
    return utt[len(prefix):] if prefix and utt.startswith(prefix) else utt


def json_safe(obj):
    """Replace non-finite floats by None so eval.json is strict JSON (json.dump would emit bare NaN)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def index_tsv(path: str, cols: list[str]) -> dict[tuple[str, str], dict]:
    """Index a critic TSV by (utt, cand), keeping only the requested numeric columns."""
    out: dict[tuple[str, str], dict] = {}
    for r in common.read_tsv(path):
        out[(r["utt"], r["cand"])] = {c: float(r[c]) for c in cols}
    LOG.info("%s: %d rows", path, len(out))
    return out


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def evaluate(rows: list[dict], mer: dict, utmos: dict, cmi: dict, cmi_gt: dict[str, float],
             utt_prefix: str = "") -> dict:
    """Aggregate the three critics over the joined rows; rows missing a critic are counted and skipped.

    Ground-truth CMI is looked up under the set utt id with ``utt_prefix`` stripped (synthesized sets).
    """
    edits = n_ref = 0
    utt_mers: list[float] = []
    utmos_vals: list[float] = []
    cmi_vals: list[float] = []
    cmi_gt_vals: list[float] = []
    dcmi_vals: list[float] = []
    miss = {"mer": 0, "utmos": 0, "cmi": 0, "cmi_gt": 0}
    for r in rows:
        key = (r["utt"], r["cand"])
        m = mer.get(key)
        if m is None:
            miss["mer"] += 1
        elif m["n_ref"] > 0:
            edits += int(m["edits"])
            n_ref += int(m["n_ref"])
            utt_mers.append(m["mer"])
        u = utmos.get(key)
        if u is None:
            miss["utmos"] += 1
        else:
            utmos_vals.append(u["utmos"])
        c = cmi.get(key)
        if c is None:
            miss["cmi"] += 1
        else:
            cmi_vals.append(c["cmi"])
            g = cmi_gt.get(gt_utt(r["utt"], utt_prefix))
            if g is None:
                miss["cmi_gt"] += 1
            else:
                cmi_gt_vals.append(g)
                dcmi_vals.append(abs(c["cmi"] - g))
    utmos_m, utmos_s = mean_std(utmos_vals)
    mer_utt_m, _ = mean_std(utt_mers)
    cmi_m, _ = mean_std(cmi_vals)
    cmi_gt_m, _ = mean_std(cmi_gt_vals)
    dcmi_m, dcmi_s = mean_std(dcmi_vals)
    return {
        "n_rows": len(rows),
        "n_utts": len({r["utt"] for r in rows}),
        "utmos_mean": utmos_m,
        "utmos_std": utmos_s,
        "n_utmos": len(utmos_vals),
        "mer_corpus_pct": 100.0 * edits / n_ref if n_ref else float("nan"),
        "mer_utt_mean_pct": 100.0 * mer_utt_m,
        "mer_edits": edits,
        "mer_n_ref": n_ref,
        "n_mer": len(utt_mers),
        "cmi_mean_pct": 100.0 * cmi_m,
        "cmi_gt_mean_pct": 100.0 * cmi_gt_m,
        "dcmi_mean_pct": 100.0 * dcmi_m,
        "dcmi_std_pct": 100.0 * dcmi_s,
        "n_dcmi": len(dcmi_vals),
        "missing": miss,
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows = load_set_rows(args.set_dir, args.cand_label)
    if args.cand is not None:
        rows = [r for r in rows if r["cand"] == args.cand]
        LOG.info("cand filter '%s': %d rows", args.cand, len(rows))
    if not rows:
        raise SystemExit("no rows to evaluate")

    mer = index_tsv(args.mer_tsv, ["edits", "n_ref", "mer"])
    utmos = index_tsv(args.utmos_tsv, ["utmos"])
    cmi = index_tsv(args.cmi_tsv, ["cmi"])
    gt_path = args.cmi_gt_tsv or args.cmi_tsv
    cmi_gt = {utt: v["cmi"] for (utt, cand), v in index_tsv(gt_path, ["cmi"]).items() if cand == "gt"}
    LOG.info("ground-truth CMI rows: %d (from %s)", len(cmi_gt), gt_path)
    if rows[0]["cand"] == "gt" and gt_path == args.cmi_tsv:
        LOG.warning("set rows are cand='gt' and compared against themselves: ΔCMI is 0 by construction "
                    "(this is the ground-truth row of Table 1)")

    res = evaluate(rows, mer, utmos, cmi, cmi_gt, args.utt_prefix)
    res["set_name"] = args.set_name or os.path.basename(os.path.normpath(args.set_dir))
    res["cand_filter"] = args.cand
    res["utt_prefix"] = args.utt_prefix
    res["inputs"] = {"set_dir": args.set_dir, "mer_tsv": args.mer_tsv, "utmos_tsv": args.utmos_tsv,
                     "cmi_tsv": args.cmi_tsv, "cmi_gt_tsv": gt_path}
    if res["missing"]["cmi_gt"] and res["n_dcmi"] == 0:
        LOG.error("no set utt matched a cand='gt' row of %s: dCMI is undefined. Synthesized set ids carry a "
                  "prefix? pass --utt_prefix (20_synthesize.py --utt_prefix)", gt_path)

    out = args.out or os.path.join(args.set_dir, "eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(json_safe(res), f, indent=2, ensure_ascii=False, allow_nan=False)  # nan -> null
    print(f"{'set':<20} {'UTMOS':>7} {'MER%':>7} {'CMI%':>7} {'CMIgt%':>7} {'dCMI%':>7} {'rows':>6}")
    print(f"{res['set_name']:<20} {res['utmos_mean']:>7.3f} {res['mer_corpus_pct']:>7.2f} {res['cmi_mean_pct']:>7.2f} "
          f"{res['cmi_gt_mean_pct']:>7.2f} {res['dcmi_mean_pct']:>7.2f} {res['n_rows']:>6}")
    if any(res["missing"].values()):
        LOG.warning("rows missing critic values: %s", res["missing"])
    LOG.info("wrote %s", out)


if __name__ == "__main__":
    main()
