export Configuration
export set_hardware

@kwdef struct Configuration{SC,SL,RT,HW}
    schemes::SC
    solvers::SL
    runtime::RT
    hardware::HW
end
Adapt.@adapt_structure Configuration

set_hardware(;backend, workgroup) = begin
    (backend=backend, workgroup=workgroup)
end
