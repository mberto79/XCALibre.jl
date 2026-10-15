using XCALibre, Test

# The OpenFOAM writer gives a field without boundary conditions of its own (density, the wall
# distance) a zero-gradient entry on every patch instead of stopping at the missing entry.
mesh = UNV3D_mesh(pkgdir(XCALibre, "examples/0_GRIDS/3d_box_1000x1000x1000mm_5.unv"), scale=0.001)
patches = (:x_min, :x_max, :y_min, :y_max, :z_min, :z_max)
BCs = assign(region=mesh, (
    U = [Dirichlet(:x_min, [1.0, 0.0, 0.0]); [Zerogradient(p) for p ∈ patches[2:end]]],
    p = [Zerogradient(p) for p ∈ patches]))
U, p, rho, y = VectorField(mesh), ScalarField(mesh), ScalarField(mesh), ScalarField(mesh)
initialise!(rho, 1.2)

mktempdir() do dir
    cd(dir) do
        writer = initialise_writer(OpenFOAM(), mesh)
        write_results(5, 5, mesh, writer, BCs, ("U", U), ("p", p), ("rho", rho), ("y", y))
        for name ∈ ("U", "p", "rho", "y")
            @test isfile(joinpath("5", name))
        end
        txt = read(joinpath("5", "rho"), String)
        bf = txt[findfirst("boundaryField", txt).stop:end]
        @test all(occursin(String(p), bf) for p ∈ patches)
        @test count("zeroGradient", bf) == length(patches)
        @test occursin("[1 -3 0 0 0 0 0]", txt)
    end
end
