using XCALibre
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using KernelAbstractions
using ILUZero

n = 5
Acsc = sprand(n,n,0.5) + I
typeof(Acsc)

P = XCALibre.Solve.Preconditioner{DILU}(Acsc)

config = (; hardware = (backend=CPU(), workgroup=32))

XCALibre.Solve.update_preconditioner!(P,  nothing, config)

Acsc
D_star = Diagonal(P.storage.D)
D_star_inv = Diagonal(1.0./P.storage.D)
L = UnitLowerTriangular(Acsc) - I
U = UnitUpperTriangular(Acsc) - I

PL = (D_star + L)*D_star_inv
PU = (D_star + U)

Acsc
Pcsc = PL*PU
b = Pcsc*ones(n)
Pcsc\b

y = PL\b
x = PU\y

x = zeros(n)
XCALibre.Solve.ldiv!(x, P.storage, b)
x