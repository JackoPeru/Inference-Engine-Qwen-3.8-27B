"""Switch E1 scripts from (1+w) to direct *w (GGUF stores direct ~1-scale norms)."""
import io
files = [
    "/home/matteo/hermes-27/scripts/e1_runner.py",
    "/home/matteo/hermes-27/scripts/e1_teacher.py",
    "/home/matteo/hermes-27/scripts/blk0_stats.py",
    "/home/matteo/hermes-27/scripts/qkv_cmp.py",
]
for fp in files:
    s = open(fp).read()
    n0 = s.count("(1+w)")
    s = s.replace("(1+w)", "w")
    n1 = s.count("((w) if plus else w)")
    s = s.replace("((w) if plus else w)", "w")
    # blk0_stats helper: *((1+w) if plus else w) already handled; also '(1+w) if plus else w' variants
    s = s.replace("(w if plus else w)", "w")
    open(fp, "w").write(s)
    print(f"{fp}: replaced {n0}+{n1}")
print("E1_NORM_PATCH_DONE")
