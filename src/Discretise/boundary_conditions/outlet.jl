export Outlet


"""
    Outlet <: AbstractNeumann

Outlet boundary condition with backflow prevention.

For the **density-based solver**, this behaves as a zero-gradient (extrapolated)
outlet when flow is leaving the domain. When backflow is detected (interior normal
velocity directed into the domain), an impermeable wall flux is applied to prevent
spurious mass ingestion. This is equivalent to a one-way valve at the boundary.

For **pressure-based solvers** (SIMPLE/PISO), this behaves identically to
`Zerogradient` and applies to both scalar and vector fields.

# Example
    Outlet(:outlet)
"""
struct Outlet{I,V,R<:UnitRange} <: AbstractNeumann
    ID::I
    value::V
    IDs_range::R
end
Adapt.@adapt_structure Outlet

Outlet(name::Symbol) = Outlet(name, 0)

@define_boundary Outlet Laplacian{Linear} begin
    0.0, 0.0
end

@define_boundary Outlet Divergence{Linear} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap, 0.0
end

@define_boundary Outlet Divergence{Upwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    max(ap, 0.0), 0.0  # clip reversed flow — see Zerogradient comment
end

@define_boundary Outlet Divergence{LUST} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    max(ap, 0.0), 0.0
end

@define_boundary Outlet Divergence{BoundedUpwind} begin
    flux = term.flux[fID]
    ap = term.sign*(flux)
    ap-flux, 0.0
end

@define_boundary Outlet Si begin
    0.0, 0.0
end
