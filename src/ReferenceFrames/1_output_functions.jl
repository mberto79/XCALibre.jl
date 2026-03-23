export save_output_polar
export get_mesh_name
export output_directory


function save_output_polar(model::Physics{T,F,SO,M,Tu,E,D,BI}, outputWriter, iteration, time, config, x0, rotaxis; mask=nothing
    ) where {T,F,SO,M,Tu,E,D,BI}
    U = model.momentum.U
    mesh = U.mesh
    cells = mesh.cells
    Up = VectorField(mesh)

    for i ∈ eachindex(Up.x.values)
        r = cells[i].centre - x0
        r_norm = r./norm(r)
        tang = r_norm × rotaxis
        Up.x.values[i] = U[i] ⋅ r_norm
        Up.z.values[i] = U.z.values[i]
        Up.y.values[i] = -(U[i] ⋅ tang)
    end
    if !isnothing(mask)
        args = (
            ("U", model.momentum.U), 
            ("Up", Up), 
            ("p", model.momentum.p),
            ("mask", mask)
        )
    else
        args = (
            ("U", model.momentum.U), 
            ("Up", Up), 
            ("p", model.momentum.p)
        )
    end
    write_results(iteration, time, model.domain, outputWriter, config.boundaries, args...)
end

function get_mesh_name(s::String)
    parts = split(s, '\\')
    part = parts[end]
    mesh = split(part, '.')
    return mesh[1]*"__"
end

function output_directory(output_dir::String, script_name::String, pattern::String="vtk", overwrite::Bool=true)
    # Make output directory
    if !isdir(output_dir)
        mkdir(output_dir)
    end

    # Find and move CFD outputs
    for file in readdir()
        if isfile(file) && occursin(pattern, file)
            mv(file, joinpath(output_dir, file); force=true)
        end
    end

    # Save a copy of input script for traceability
    source = script_name
    base = splitext(basename(source))[1]   # filename without extension
    dest = joinpath(output_dir, base * ".txt")
    cp(source, dest; force=overwrite)
end