n = 20
Acsc = sprand(n,n,0.1) + 0.75I

i, j, v = findnz(Acsc)
Acsr = sparsecsr(i, j, v, n, n)

P = XCALibre.Solve.Preconditioner{DILU}(Acsr)

config = (; hardware = (backend=CPU(), workgroup=32))

XCALibre.Solve.update_preconditioner!(P,  nothing, config)

D_star = Diagonal(P.storage.D)
D_star_inv = Diagonal(1.0./P.storage.D)
L = UnitLowerTriangular(Acsc) - I
U = UnitUpperTriangular(Acsc) - I

PL = (D_star + L)*D_star_inv
PU = (D_star + U)

Pcsc = PL*PU
@test diag(Pcsc) ≈ diag(Acsc)
b = Pcsc*ones(n)

y = PL\b
x = PU\y

yDILU = zeros(n)
XCALibre.Solve.forward_substitution!(yDILU, P.storage, b)
@test yDILU ≈ y

xDILU = zeros(n)
XCALibre.Solve.backward_substitution!(xDILU, P.storage, yDILU)
@test xDILU ≈ x 

xLDIV = zeros(n)
XCALibre.Solve.ldiv!(xLDIV, P.storage, b)
@test xLDIV ≈ x