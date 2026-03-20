export save_postprocessing

function save_postprocessing(postprocess, iteration, time, mesh, meshData, BCs)
    postprocess === nothing && return nothing
    # suffix = "_" * string(postprocess.name)  
    suffix = "_postprocessed"
    
    args = build_args(postprocess)
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix=suffix)
end
function save_postprocessing(postprocess, iteration, time, mesh, meshData::FOAMWriter, BCs)
    postprocess === nothing && return nothing
    args = build_args(postprocess)
    write_results(iteration, time, mesh, meshData, BCs, args...; suffix="")
end

function build_args(pp)
    pp === nothing && return ()
    if hasproperty(pp, :rs)
        return ((getproperty(pp, :name), getproperty(pp, :rs)),)
    elseif hasproperty(pp, :rms)
        return ((getproperty(pp, :name), getproperty(pp, :rms)),)
    elseif hasproperty(pp, :mean)
        return ((getproperty(pp, :name), getproperty(pp, :mean)),)
    else
        return ()
    end
end

function build_args(pp::Vector)
    vector_of_tuples = build_args.(pp)
    return Tuple(first.(vector_of_tuples))
end