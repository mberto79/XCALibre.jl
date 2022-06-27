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