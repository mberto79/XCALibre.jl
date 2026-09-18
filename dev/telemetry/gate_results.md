# Gate results

| Timestamp | Session # | Gate Name | Target Metric | Actual Result | Status | Commit |
|---|---:|---|---|---|---|---|
| 2026-09-17 | 0 | serial suite | all pass | 1544/1544 pass, exit 0 | PASS | 9ca6f2c2 |
| 2026-09-17 | 0 | distributed suite n=1,2 | all pass | 26/26 pass | PASS | 9ca6f2c2 |
| 2026-09-17 | 1 | distributed gate n=2 | 5 files pass within 300 s | 5/5 pass, 89.3 s wall | PASS | 3887cc8b |
| 2026-09-17 | 2 | documentation build | builds with no errors | 0 errors, distributed page in output | PASS | 99d03e04 |
| 2026-09-17 | 3 | serial suite (full, Pkg.test) | all pass, no reduction on 1544 | 1548 pass / 1 fail: test_halo.jl `using Random` undeclared in test/Project.toml | FAIL -> fixed, rerun pending | 101c5b4b |
