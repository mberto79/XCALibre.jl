export save_postprocessing

function save_postprocessing(postprocess, iteration, time, mesh, meshData, BCs, args...)
    postprocess === nothing && return nothing
    suffix = "_" * string(postprocess.field)  
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix=suffix)
end

function save_postprocessing(postprocess, iteration, time, mesh, meshData::FOAMWriter, BCs, args...)
    postprocess === nothing && return nothing
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix=nothing)
end