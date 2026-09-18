# Freeze residual cost, distributed backward-facing-step, n=4, 499,503 cells, 100 SIMPLE iterations

Probe: `dev/scripts/scaling_probe.jl worker <partdir_n4> pc=<pc> reuse=<N> 100` on stock binaries (`dev/petscenv_stock`), 2026-09-18. Clock UNPINNED (~3600 MHz): the s/iter column is not timing evidence (use `scaling.csv`); the residuals are deterministic at fixed rank count and are the measurement here.

```text
PROBE pc=gamg reuse=1 nranks=4 ncells=499503 t3=1.839 t100=32.901 per_iter=0.3202 mhz=3700.0 p=0.0002953331642304737 Ux=0.0006236456822006411 Uy=0.00972286054080918 Uz=0.011790049148953432
PROBE pc=gamg reuse=10 nranks=4 ncells=499503 t3=1.697 t100=25.51 per_iter=0.2455 mhz=3610.8 p=0.000294987660638364 Ux=0.0006234056673180801 Uy=0.009680157456455294 Uz=0.011805643918548245
PROBE pc=gamg reuse=25 nranks=4 ncells=499503 t3=1.652 t100=24.896 per_iter=0.2396 mhz=3600.0 p=0.00029542235798841023 Ux=0.0006232872550599325 Uy=0.009677581873496141 Uz=0.011803714013716651
PROBE pc=boomeramg reuse=10 nranks=4 ncells=499503 t3=1.748 t100=33.476 per_iter=0.3271 mhz=3601.8 p=7.825560785715877e-5 Ux=0.0006224674115455534 Uy=0.009658312701130354 Uz=0.011750055654586426
PROBE pc=boomeramg reuse=25 nranks=4 ncells=499503 t3=1.725 t100=34.149 per_iter=0.3343 mhz=3600.0 p=0.00018466243571015474 Ux=0.000622810341321781 Uy=0.009662992922598023 Uz=0.011756733434231705
```
