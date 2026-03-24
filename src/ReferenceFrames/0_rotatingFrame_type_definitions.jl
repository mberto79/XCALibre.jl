export RotatingFrame
export RotatingFrames2D

struct RotatingFrames2D
    frames
    global_mask
end

RotatingFrames2D(; hardware, mesh, Frames) = begin
    ID = 1
    n = length(Frames)
    Omega = zeros(Float64, n)
    Rotaxis = Vector{Vector{Float64}}(undef, n)
    X0 = Vector{Vector{Float64}}(undef, n)
    X1 = Vector{Vector{Float64}}(undef, n)
    global_mask = ScalarField(mesh)
    for frame in Frames
        (; omega, rotaxis, x0, x1, radius_inner, radius_outer) = frame
        Omega[ID] = omega
        Rotaxis[ID] = rotaxis
        X0[ID] = x0
        X1[ID] = x1
        mask = radial_mask(x0, radius_inner, radius_outer, hardware, mesh) * ID
        global_mask = global_mask + mask
        ID = ID+1
    end

    frames = (omega, rotaxis, x0)
    RotatingFrames2D(frames, global_mask)
end


struct RotatingFrameStruct
    omega
    rotaxis
    x0
    x1
    radius_inner
    radius_outer
    mask
end

RotatingFrame(; omega, rotaxis=nothing, x0=nothing, x1=nothing, radius_inner::Float64=0.0, radius_outer::Float64, hardware, mesh) = begin
    mask = radial_mask(x0,  radius_inner, radius_outer, hardware, mesh)

    if isnothing(rotaxis)
        if isnothing(x1)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        elseif isnothing(x0)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        end
        rotaxis = x1 - x0
    elseif isnothing(x0)
        if isnothing(x1)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        elseif isnothing(rotaxis)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        end
        x0 = x1 - rotaxis
    elseif isnothing(x1)
        if isnothing(x0)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        elseif isnothing(rotaxis)
            println("Error: You either need to define two points, x1 and x0, or a point and a rotation vector, x0 and rotaxis.")
        end
        x1 = x0 + rotaxis
    end

    rotaxis = SVector{3}(rotaxis) 
    x0 = SVector{3}(x0)
    x1 = SVector{3}(x1)

    RotatingFrameStruct(
    omega,
    rotaxis,
    x0,
    x1,
    radius_inner,
    radius_outer,
    mask
    )
end