export surface_flux!, surface_normal_gradient!, surface_normal_gradient2!

# surface_flux(snflux, facesID, cellsID, phi) = begin
surface_flux!(snflux, phi, IDs_range) = begin
    mesh = phi.mesh
    (; faces, boundary_cellsID) = mesh

    (; cells, faces) = mesh
    for i ∈ eachindex(snflux.x)
        # cID = cellsID[i]
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
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

surface_normal_gradient!(snGrad2, U, Uf, IDs_range) = begin
    mesh = U.mesh
    (; faces, boundary_cellsID) = mesh
    for i ∈ eachindex(snGrad2)
        fID = IDs_range[i]
        cID = boundary_cellsID[fID]
        face = faces[fID]
        Uw = Uf[fID]
        (; normal, delta) = face
        Ui = U[cID]
        Udiff = (Ui - Uw)
        Up = Udiff - (Udiff⋅normal)*normal # parallel velocity difference
        grad = Up/(delta)
        snGrad2.x[i] = grad[1]
        snGrad2.y[i] = grad[2]
        snGrad2.z[i] = grad[3] 
        nothing
    end
end

surface_normal_gradient2!(snGrad3, U, IDs_range, config) = begin
    mesh = U.mesh
    TF = _get_float(mesh)
    (; faces, boundary_cellsID) = mesh
    ∇U = Grad{Midpoint}(U)
    Uf = FaceVectorField(U.mesh)
    (; boundaries) = config
    grad!(∇U,Uf,U,boundaries.U, zero(TF),config)
    for i ∈ eachindex(snGrad3)
        fID = IDs_range[i]
        face = faces[fID]
        (; normal) = face
        snGrad3.x[i] = ∇U[i][1]*normal[1]
        snGrad3.y[i] = ∇U[i][2]*normal[2]
        snGrad3.z[i] = ∇U[i][3]*normal[3]
    end
end