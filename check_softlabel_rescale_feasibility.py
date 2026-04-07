import argparse
import os
import sys
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Check whether solution ranges are reachable under current _rescale.py hard constraints.")
    p.add_argument("--repo-root", required=True)
    p.add_argument("--solution", required=True, help="Path to solution.npz")
    return p.parse_args()


def main():
    args = parse_args()
    util_dir = os.path.join(args.repo_root, "pinn_spm_param", "util")
    if util_dir not in sys.path:
        sys.path.insert(0, util_dir)

    from spm import makeParams  # type: ignore

    params = makeParams()
    z = np.load(args.solution)

    cs_a = np.asarray(z["cs_a"], dtype=np.float64)
    cs_c = np.asarray(z["cs_c"], dtype=np.float64)

    cs_a0 = float(params["cs_a0"])
    cs_c0 = float(params["cs_c0"])
    csanmax = float(params["csanmax"])
    cscamax = float(params["cscamax"])

    # current _rescale.py reachability
    cs_a_min_reach = 0.0
    cs_a_max_reach = cs_a0
    cs_c_min_reach = cs_c0
    cs_c_max_reach = cscamax

    print("Current hard-rescale reachable intervals under _rescale.py:")
    print(f"  cs_a reachable: [{cs_a_min_reach:.12g}, {cs_a_max_reach:.12g}]")
    print(f"  cs_c reachable: [{cs_c_min_reach:.12g}, {cs_c_max_reach:.12g}]")
    print("Soft-label intervals from solution.npz:")
    print(f"  cs_a labels   : [{cs_a.min():.12g}, {cs_a.max():.12g}]")
    print(f"  cs_c labels   : [{cs_c.min():.12g}, {cs_c.max():.12g}]")

    ok_a = (cs_a.min() >= cs_a_min_reach - 1e-12) and (cs_a.max() <= cs_a_max_reach + 1e-12)
    ok_c = (cs_c.min() >= cs_c_min_reach - 1e-12) and (cs_c.max() <= cs_c_max_reach + 1e-12)

    print("Feasibility:")
    print(f"  cs_a reachable by current hard transform? {'YES' if ok_a else 'NO'}")
    print(f"  cs_c reachable by current hard transform? {'YES' if ok_c else 'NO'}")

    if not ok_a or not ok_c:
        print("Conclusion: current data-only training is structurally constrained by _rescale.py.")
        print("For exact soft-label fitting, switch to identity/symmetric output rescaling for cs_a/cs_c.")


if __name__ == "__main__":
    main()
