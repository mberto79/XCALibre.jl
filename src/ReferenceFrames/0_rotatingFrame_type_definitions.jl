export REF_FRAME

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
