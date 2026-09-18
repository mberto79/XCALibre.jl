# Gate results

| Timestamp | Session # | Gate Name | Target Metric | Actual Result | Status | Commit |
|---|---:|---|---|---|---|---|
| 2026-09-17 | 0 | serial suite | all pass | 1544/1544 pass, exit 0 | PASS | 9ca6f2c2 |
| 2026-09-17 | 0 | distributed suite n=1,2 | all pass | 26/26 pass | PASS | 9ca6f2c2 |
| 2026-09-17 | 1 | distributed gate n=2 | 5 files pass within 300 s | 5/5 pass, 89.3 s wall | PASS | 3887cc8b |
| 2026-09-17 | 2 | documentation build | builds with no errors | 0 errors, distributed page in output | PASS | 99d03e04 |
| 2026-09-17 | 3 | serial suite (full, Pkg.test) | all pass, no reduction on 1544 | 1548 pass / 1 fail: test_halo.jl `using Random` undeclared in test/Project.toml | FAIL -> fixed, rerun pending | 101c5b4b |
| 2026-09-18 | 3 | serial suite (full, Pkg.test) | all pass, no reduction on 1544 | 1549/1549 pass, exit 0, 21m15s | PASS | 2374a1e7 |
| 2026-09-18 | 3 | distributed gate n=2 (under Pkg.test) | 5 files pass within 300 s | 5/5 pass, 186.7 s at a pinned 2200 MHz | PASS | 2374a1e7 |
| 2026-09-18 | 3 | example on stock binaries, clean dir | runs at n=2 and n=4, writes decomposed output | parts/, processor0-3/, XCALibre.foam at both | PASS | 101c5b4b |
