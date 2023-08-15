export surface_flux, surface_normal_gradient

surface_flux(snflux, facesID, cellsID, phi) = begin
    mesh = phi.mesh
    (; cells, faces) = mesh
    for i ∈ eachindex(snflux.x)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        (; area, normal) = face
        Sf = area*normal
        flux = phi[cID]*Sf
        snflux.x[i] = flux[1]
        snflux.y[i] = flux[2]
        snflux.z[i] = flux[3] 
        nothing
    end
end

surface_normal_gradient(snGrad, facesID, cellsID, U, Uw) = begin
    mesh = U.mesh
    (; faces) = mesh
    for i ∈ eachindex(snGrad)
        cID = cellsID[i]
        fID = facesID[i]
        face = faces[fID]
        (; normal, delta) = face
        Ui = U[cID]
        Udiff = (Ui - Uw)
        Up = Udiff - (Udiff⋅normal)*normal # oarallel velocity difference
        grad = Up/delta
        snGrad.x[i] = grad[1]
        snGrad.y[i] = grad[2]
        snGrad.z[i] = grad[3] 
        nothing
    end
end