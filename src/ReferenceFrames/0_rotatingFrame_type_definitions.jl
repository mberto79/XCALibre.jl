export RotatingFrame
export RotatingFrames

struct RotatingFrames
    frames
    global_mask
end

RotatingFrames(; hardware, mesh, Frames) = begin
    ID = 1
    omega = []
    rotaxis = []
    x0 = []
    global_mask - ScalarField(mesh)
    for frame in Frames
        (; omega, rotaxis, x0, radius_inner, radius_outer) = frame
        omega[ID] = omega
        rotaxis[ID] = rotaxis
        x0[ID] = x0
        mask = radial_mask(x0, radius_inner, radius_outer, hardware, mesh) * ID
        global_mask = global_mask + mask
        ID = ID+1
    end

    frames = (omega, rotaxis, x0)
    RotatingFrames(frames, global_mask)
end


struct RotatingFrameStruct
    omega
    rotaxis
    x0
    radius_inner
    radius_outer
    mask
end

RotatingFrame(; omega, rotaxis, x0, radius_inner::Float64=0.0, radius_outer::Float64, hardware, mesh) = begin
    mask = radial_mask(x0,  radius_inner, radius_outer, hardware, mesh)
    rotaxis = SVector{3}(rotaxis) 
    x0 = SVector{3}(x0)

    RotatingFrameStruct(
    omega,
    rotaxis,
    x0,
    radius_inner,
    radius_outer,
    mask
    )
end