using XCALibre

grids_dir = pkgdir(XCALibre, "Test_Meshes/");
grid = "3x4_1m_grid.unv";
mesh_file = joinpath(grids_dir, grid);
output=VTK()

mesh = UNV2D_mesh(mesh_file, scale=0.001);

outputWriter = initialise_writer(output, mesh)

backend = CPU();

Cids = ScalarField(mesh)
for i ∈ eachindex(mesh.cells)
    Cids[i] = i
end

hardware = Hardware(backend=backend, workgroup=1024);

bc = assign(
    region=mesh,
    (
    A = [
        Extrapolated(:Left),
        Extrapolated(:Right),
        Extrapolated(:Top),
        Extrapolated(:Bottom)
    ],
    B = [
        Dirichlet(:Left,0),
        Dirichlet(:Right,0),
        Dirichlet(:Top,0),
        Dirichlet(:Bottom,0)
    ],
    C = [
        Dirichlet(:Left,[0,0,0]),
        Dirichlet(:Right,[0,0,0]),
        Dirichlet(:Top,[0,0,0]),
        Dirichlet(:Bottom,[0,0,0])
    ],
    D = [
        Dirichlet(:Left,[1,0,0]),
        Dirichlet(:Right,[1,0,0]),
        Dirichlet(:Top,[0,1,0]),
        Dirichlet(:Bottom,[0,1,0])
    ],
    E = [
        Dirichlet(:Left,1),
        Dirichlet(:Right,1),
        Dirichlet(:Top,1),
        Dirichlet(:Bottom,1)
    ],
    F = [
        Dirichlet(:Left, 0),
        Extrapolated(:Right),
        Extrapolated(:Bottom),
        Dirichlet(:Top, 1)
    ]
    )
)


config = Configuration(
    solvers=empty, schemes=empty, runtime=empty, hardware=hardware, boundaries=empty);


values = ScalarField(mesh)
valuesf = FaceScalarField(mesh)
laplacian = ScalarField(mesh)

for i ∈ eachindex(mesh.cells)
    if (abs(mesh.cells[i].centre[1]-0.5) < 1e-3) && (abs(mesh.cells[i].centre[2]-0.5) < 1e-3)
        values[i] = 1
        continue
    end
    values[i] = 0

end

begin
values[1] = 2
values[2] = 1
values[3] = 0
values[4] = 3
values[5] = 2
values[6] = 1
values[7] = 4
values[8] = 3
values[9] = 2
values[10] = 5
values[11] = 4
values[12] = 3
end

interpolate!(valuesf, values, config)
#correct_boundaries!(valuesf, values, bc.A, 0, config)
#correct_boundaries!(valuesf, values, bc.B, 0, config)
correct_boundaries!(valuesf, values, bc.F, 0, config)

grad_val=Grad{Gauss}(valuesf)

grad!(grad_val, valuesf, config)
grad_valf = FaceVectorField(mesh)
interpolate!(grad_valf, grad_val.result, config)
correct_boundaries!(grad_valf, grad_val.result, bc.C, 0, config)
lap_val = ScalarField(mesh)
div!(lap_val, grad_valf, config)

vec_field = VectorField(mesh)
initialise!(vec_field, [0,0,0])
lap_val_2 = ScalarField(mesh)
for (i,boundary) ∈ enumerate(mesh.boundaries)
    (; IDs_range) = boundary
    bcellID = mesh.boundary_cellsID[IDs_range]
    for i ∈ eachindex(IDs_range)
        fcentre = mesh.faces[IDs_range[i]].centre
        ccentre = mesh.cells[bcellID[i]].centre
        Δpos = 2 .*(fcentre.-ccentre)
        if abs(Δpos[1]) > 1e-10
            vec_field.x.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[1])*(valuesf[IDs_range[i]]-values[bcellID[i]])/abs2(Δpos[1])
        end
        if abs(Δpos[2]) > 1e-10
            vec_field.y.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[2])*(valuesf[IDs_range[i]]-values[bcellID[i]])/abs2(Δpos[2])
        end
        if abs(Δpos[3]) > 1e-10
            vec_field.z.values[bcellID[i]] += 2*abs(mesh.faces[IDs_range[i]].normal[3])*(valuesf[IDs_range[i]]-values[bcellID[i]])/abs2(Δpos[3])
        end
        println("$(fcentre), $(ccentre), $(Δpos), $(valuesf[IDs_range[i]]), $(values[bcellID[i]]), $(vec_field.x.values[bcellID[i]])")
    end
end
for i ∈ eachindex(mesh.cells)
    (; faces_range, centre) = mesh.cells[i]
    main_centre = centre
    x_y_vals = [0,0,0]

    for fID ∈ mesh.cell_faces[faces_range]
        (; normal, ownerCells, delta) = mesh.faces[fID]
        for cellID ∈ ownerCells
            (; centre) = mesh.cells[cellID]
            Δpos = centre.-main_centre
            
            if abs(Δpos[1]) > 1e-10
                vec_field.x[i] += abs(mesh.faces[fID].normal[1])*(values[cellID]-values[i])/abs2(Δpos[1])
            end
            if abs(Δpos[2]) > 1e-10
                vec_field.y[i] += abs(mesh.faces[fID].normal[2])*(values[cellID]-values[i])/abs2(Δpos[2])
            end
            if abs(Δpos[3]) > 1e-10
                vec_field.z[i] += abs(mesh.faces[fID].normal[3])*(values[cellID]-values[i])/abs2(Δpos[3])
            end
        end
    end

end
@. lap_val_2.values = vec_field.x.values + vec_field.y.values + vec_field.z.values

#laplacian!(laplacian, valuesf, values, bc.B, 0, config)

div_test_initial = VectorField(mesh)
div_test_initialf = FaceVectorField(mesh)
div_test = ScalarField(mesh)
for i ∈ eachindex(mesh.cells)
    div_test_initial.x[i] = 1
    div_test_initial.y[i] = 1
end
interpolate!(div_test_initialf, div_test_initial, config)
correct_boundaries!(div_test_initialf, div_test_initial, bc.C, 0, config)
div!(div_test, div_test_initialf, config)

args = (
    ("vals", values),
    ("a", lap_val),
    ("b", lap_val_2),#laplacian),
    ("Cid", Cids),
    ("grad", grad_val.result),
    #("div_init", div_test_initial),
    #("div", div_test),
    ("vec_field", vec_field),
    ("lapl", lap_val_2)

)
write_results(1, 1, mesh, outputWriter, bc, args...)