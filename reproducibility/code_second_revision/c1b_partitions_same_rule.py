"""Repeated group-disjoint partitions under the rule of the primary partitions.

c1_partitions.py resamples the assignment by holding out whole groups of any
size. The primary scaffold-, cluster- and source-disjoint partitions were built
differently: the hold-out is filled with the smallest groups first, in a seeded
random order among groups of equal size (a4_splits.group_disjoint_split), so
it consists of structurally isolated compounds, 191 singleton scaffolds or 191
singleton clusters, and of the smallest publications. Holding out isolated
compounds and holding out whole analogue families are different tests, and a
result that appears under one need not appear under the other.

This script therefore also repeats the primary rule itself, with five further
seeds. It is the like-for-like replicate of the primary partitions, with one
caveat that is reported with the results: because the hold-out is drawn from a
limited pool (363 singleton scaffolds, 290 singleton clusters, and for the
source design the same 128 compounds every time), partitions drawn under this
rule share much of their hold-out with each other and with the primary one, so
they are far from independent. The overlap is computed and stored here.

The scaffold- and cluster-disjoint designs are replicated (partition ids
scaffold_s01 ... cluster_s05, appended to c1_partitions.json so that the same
worker scripts can run them). The source-disjoint design is not: under this
rule nine tenths of its hold-out is the same compounds in every draw, so a
replicate would repeat the primary partition rather than test it. Its overlap
is stored as the documented reason.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from c1_partitions import characterise, fingerprints
from common import OUT, R1_OUT, SAME_RULE_DESIGNS, SAME_RULE_SEEDS, load_light, read_json, write_json

SEEDS = SAME_RULE_SEEDS
TEST_FRACTION = 0.10


def group_disjoint_split(groups, seed, test_fraction=TEST_FRACTION):
    """Identical to a4_splits.group_disjoint_split of the first revision."""
    from collections import defaultdict

    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[g].append(i)
    order = sorted(buckets.items(), key=lambda kv: (len(kv[1]), rng.random()))
    target = int(round(test_fraction * len(groups)))
    test = []
    for _name, idx in order:
        if len(test) + len(idx) > target and test:
            continue
        test.extend(idx)
        if len(test) >= target:
            break
    test = np.array(sorted(test))
    train = np.array(sorted(set(range(len(groups))) - set(test.tolist())))
    return train, test


def main():
    smiles, y, curated, _ = load_light()
    payload = read_json(os.path.join(OUT, "c1_partitions.json"))
    primary = read_json(os.path.join(R1_OUT, "a4_splits.json"))["splits"]
    fps = fingerprints(smiles)
    scaffolds = curated["bemis_murcko_scaffold"].to_numpy()          # as in a4_splits.py
    groupings = {
        "scaffold": scaffolds,
        "cluster": np.asarray(payload["butina_cluster_of_compound"]),
        "source": curated["source_doi"].to_numpy(),
    }

    # the function reproduces the primary partitions exactly when given their seed
    for design, groups in groupings.items():
        _, te = group_disjoint_split(groups, 42)
        assert sorted(te.tolist()) == sorted(primary[design]["test_idx"]), design
    print("the rule reproduces the three primary group-disjoint partitions with seed 42")

    rows, overlaps = [], {}
    for design, groups in groupings.items():
        size_of = Counter(pd.Series(groups).astype(str))
        prim = set(primary[design]["test_idx"])
        tests = []
        for seed in SEEDS:
            tr, te = group_disjoint_split(groups, seed)
            pid = f"{design}_s{seed:02d}"
            if design in SAME_RULE_DESIGNS:
                payload["partitions"][pid] = {"design": design, "rule": "smallest groups first", "seed": seed,
                                              "train_idx": tr.tolist(), "test_idx": te.tolist()}
            else:
                payload["partitions"].pop(pid, None)
            c = characterise(fps, pd.Series(scaffolds).fillna("").astype(str).to_numpy(), y, tr, te)
            held = [size_of[str(g)] for g in set(pd.Series(groups).astype(str).to_numpy()[te])]
            c.update({"partition": pid, "design": design, "seed": seed, "replicated": design in SAME_RULE_DESIGNS,
                      "n_test_groups": len(held), "largest_test_group": int(max(held)),
                      "pct_shared_with_primary": 100 * len(prim & set(te.tolist())) / len(te)})
            rows.append(c)
            tests.append(set(te.tolist()))
        pair = [100 * len(a & b) / len(a) for i, a in enumerate(tests) for b in tests[i + 1:]]
        overlaps[design] = {"mean_pct_shared_between_two_repeats": float(np.mean(pair)),
                            "mean_pct_shared_with_primary": float(np.mean([r["pct_shared_with_primary"]
                                                                           for r in rows if r["design"] == design])),
                            "n_compounds_ever_held_out": int(len(set().union(*tests)))}
        print(f"{design:9s} hold-outs of {sorted({len(t) for t in tests})} compounds; "
              f"{overlaps[design]['mean_pct_shared_between_two_repeats']:.0f}% shared between two repeats, "
              f"{overlaps[design]['mean_pct_shared_with_primary']:.0f}% with the primary partition")

    payload["same_rule"] = {"seeds": list(SEEDS), "rule": "smallest groups first, as in a4_splits.py",
                            "designs_replicated": list(SAME_RULE_DESIGNS), "overlap": overlaps}
    write_json(os.path.join(OUT, "c1_partitions.json"), payload)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT, "c1b_same_rule_characterisation.csv"), index=False)
    print(table.groupby("design", sort=False)[["n_test", "nn_tanimoto_median", "pct_scaffold_seen",
                                               "largest_test_group", "pct_shared_with_primary"]]
          .agg(["min", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
