export RotatingFrame
export RotatingFrames2D

struct RotatingFrames2D{Data,Mask}
    frames::Data
    global_mask::Mask
end
Adapt.@adapt_structure RotatingFrames2D

RotatingFrames2D(; hardware, mesh, frames) = begin
    (; backend) = hardware
    ID = 1
    n = length(frames)
    Omega = KernelAbstractions.zeros(backend, Float64, n)
    Rotaxis = KernelAbstractions.allocate(backend, SVector{3, Float64}, n)
    X0 = KernelAbstractions.allocate(backend, SVector{3, Float64}, n)
    X1 = KernelAbstractions.allocate(backend, SVector{3, Float64}, n)
    global_mask = ScalarField(mesh)

    for frame in frames
        (; omega, rotaxis, x0, x1, radius_inner, radius_outer) = frame
        Omega[ID] = omega
        Rotaxis[ID] = rotaxis
        X0[ID] = x0
        X1[ID] = x1
        global_mask = radial_mask!(x0, radius_inner, radius_outer, hardware, mesh; ID=ID, mask=global_mask) 
        ID = ID+1
    end

    Frames = FramesData(Omega, Rotaxis, X0)
    RotatingFrames2D(Frames, global_mask)
end

struct FrameData
    omega::Array
    rotaxis::Array
    x0::Array
end
Adapt.@adapt_structure FrameData

FramesData(omega,rotaxis,x0) = begin
    FrameData(omega,rotaxis,x0)
end

struct RotatingFrame
    omega::Float64
    rotaxis::SVector
    x0::SVector
    x1::SVector
    radius_inner::Float64
    radius_outer::Float64
end
Adapt.@adapt_structure RotatingFrame

RotatingFrame(; omega, rotaxis=nothing, x0=nothing, x1=nothing, radius_inner::Float64=0.0, radius_outer::Float64, hardware, mesh) = begin
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

    RotatingFrame(
    omega,
    rotaxis,
    x0,
    x1,
    radius_inner,
    radius_outer
    )
end