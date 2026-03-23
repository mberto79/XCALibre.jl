export RotatingFrame
#export reference_frames
#
#struct reference_frames
#    Frames
#end

#refFrames(Frames) = begin
#    ID = 1
#    for frames in Frames:
#        frames +
#        ID = ID+1
#    end
#
#    reference_frames(Frames)
#end

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

#struct TrueRotatingFrameStruct
#    omega
#    rotaxis
#    x0
#    radius_inner
#    radius_outer
#    mask
#end
#
#TrueRotatingFrame(Frame) = begin
#    mask = radial_mask(x0,  radius_inner, radius_outer, hardware, mesh)
#    rotaxis = SVector{3}(rotaxis) 
#    x0 = SVector{3}(x0)
#
#    TrueRotatingFrameStruct(
#    omega,
#    rotaxis,
#    x0,
#    radius_inner,
#    radius_outer,
#    mask
#    )
#end