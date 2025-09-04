export save_postprocessing

function save_postprocessing(
    field_name, iteration, time, mesh, meshData, BCs, args...)
    
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix="_"*field_name) 
end

function save_postprocessing(
    field_name, iteration, time, mesh, meshData::FOAMWriter, BCs, args...)

    field_name = nothing # not sure it's needed please check, if not needed remove this line
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix=nothing)
end