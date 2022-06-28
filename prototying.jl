struct Laplacian{S,F,P}
    J::F
    phi::P
end
Laplacian{S}(J::F, phi::P) where {S,F,P} = Laplacian{S,F,P}(J,phi)

struct Linear end

n = 20
J = zeros(n)
phi = zeros(n)

lap = Laplacian{Linear}(J, phi)

ex = :(Laplacian{Linear}(J, phi) - Laplacian{Linear}(J,phi) = -∇p.x)

macro equation(ex)
    print(ex)
    return :(ex)
end

ex = @equation ux_eqn(
    Laplacian{Linear}(J, phi) 
    - Laplacian{Linear}(J,phi) 
    == 
    -∇p.x
    )

using Plots

β = 1
L = 0.1
N = 10
x₀ = 1.0

η(i, N) = (i-1)/(N-1)
x(i, N, β, x₀, L) = x₀ + (L/2)*(1.0 - tanh(β*(1-2*η(i,N)))/tanh(β))

i = [1:N;]
xc = x.(i,N,β,x₀, L)

scatter(i, xc)
