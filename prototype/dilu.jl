using XCALibre
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR
using KernelAbstractions
using ILUZero

n = 20
Acsc = sprand(n,n,0.1) + 0.75I
typeof(Acsc)

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
diag(Pcsc) ≈ diag(Acsc)
b = Pcsc*ones(n)
Pcsc\b

y = PL\b
x = PU\y

yDILU = zeros(n)
@time XCALibre.Solve.forward_substitution!(yDILU, P.storage, b)
yDILU
yDILU ≈ y

for i in eachindex(y); println("y = $(y[i]), ydilu = $(yDILU[i])"); end

xDILU = zeros(n)
@time XCALibre.Solve.backward_substitution!(xDILU, P.storage, yDILU)
# @time XCALibre.Solve.backward_substitution!(xDILU, P.storage, y)
xDILU
xDILU ≈ x 

for i in eachindex(y); println("x = $(x[i]), xdilu = $(xDILU[i])"); end

xldiv = zeros(n)
@time XCALibre.Solve.ldiv!(xldiv, P.storage, b)
xldiv ≈ x


XCALibre.Solve.is_ldiv(P)