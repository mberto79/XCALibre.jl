export REF_FRAME
export RotatingFrame

struct REF_FRAME_values
    type
    omega
    rotaxis
    x0
    mask
end

REF_FRAME(type, omega, rotaxis, x0, mask) = begin
    REF_FRAME_values(
        type,
        omega,
        rotaxis,
        x0,
        mask
    )
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