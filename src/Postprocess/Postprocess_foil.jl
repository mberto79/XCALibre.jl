export foil_obj_func
export paraview_vis

foil_obj_func(patch::Symbol, p::ScalarField, U::VectorField, rho, ν, turb_model,Fv=[0,0,0]) = begin
    Fp = pressure_force(patch, p, rho)
    try 
        Fv = viscous_force(patch, U, rho, ν, turb_model.nut)
    catch
        println("\nNo viscous force component used due to function error!")
    end
    Ft = Fp + Fv
    aero_eff = Ft[2]/Ft[1]
    print("\nAerofoil L/D: ",round(aero_eff,sigdigits = 4))
    return aero_eff
end

paraview_vis(;paraview_path::String,vtk_path::String) = begin
    try
        run(`"$paraview_path" "$vtk_path"`)
    catch
        println("\nIncorrect pointer to paraview installation!")
    end
end