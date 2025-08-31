using XCALibre

include("../src/Preprocess/setFields.jl")

grids_dir = pkgdir(XCALibre, "examples/0_GRIDS")
grid = "quad40.unv"
mesh_file = joinpath(grids_dir, grid)

mesh = UNV2D_mesh(mesh_file)

backend = CPU(); workgroup = AutoTune(); activate_multithread(backend)

hardware = Hardware(backend=backend, workgroup=workgroup)
mesh_dev = adapt(backend, mesh)

# Setup dummy model
model = Physics(
    time = Steady(),
    fluid = Fluid{Incompressible}(nu = 1.0e-3),
    turbulence = RANS{Laminar}(),
    energy = Energy{Isothermal}(),
    domain = mesh_dev
    )



## DOMAIN FOR TESTING - quad40.unv
## It is a square in X-Y plane spanning from (0,0,0) to (1000,1000,0)


## BOX FIELD TEST

# Aim to change 4 cells at the domain corner at (1000, 1000, 0)
# dx=dy=25
# thus cells are in the range (950, 950, 0) to (1000, 1000, 0) - this would definitely include their centres

modified_cell_value_expected = 1.0
unmodified_cell_value_expected = 0.0

box_test_expected_cells_amount = 4

initialise!(model.momentum.p, 0.0)
box_test_modified_cells_amount = setField_Box!(mesh=mesh, field=model.momentum.p, value=1.0, min_corner=[950.0, 950.0, 0.0], max_corner=[1000.0, 1000.0, 0.0])

box_test_random_modified_cell_value = model.momentum.p[1]
box_test_random_unmodified_cell_value = model.momentum.p[100]


@test box_test_modified_cells_amount ≈ box_test_expected_cells_amount
@test box_test_random_modified_cell_value ≈ modified_cell_value_expected
@test box_test_random_unmodified_cell_value ≈ unmodified_cell_value_expected


## 2D CIRCLE FIELD TEST

# We put the centre of a cirlce at (1050, 1050)
# Cells close to the edge at (1000,1000) are of our interest
# We can calculate distances from circle's centre to the nearest cells:
#       COORDINATES <> DISTANCE
# 1.    (987.5, 987.5) <> 88.39
# 2.    (962.5, 962.5) <> 123.74
# 3.    (962.5, 987.5) <> 107.53
# 4.    (937.5, 987.5) <> 128.70

# If the radius is 100, then only 1 cell at the edge is captured (case #1)
# If the radius is 115, then 3 cells are captured (cases #1 and #3)
# If the radius is 125, then 4 cells are captured (cases #1,#2,#3, while it should not capture #4)


circle_test1_expected_cells_amount = 1
circle_test2_expected_cells_amount = 3
circle_test3_expected_cells_amount = 4

initialise!(model.momentum.p, 0.0)
circle_test1_modified_cells_amount = setField_Circle2D!(mesh=mesh, field=model.momentum.p, value=1.0, centre=[1050.0,1050.0], radius=100.0)

initialise!(model.momentum.p, 0.0)
circle_test2_modified_cells_amount = setField_Circle2D!(mesh=mesh, field=model.momentum.p, value=1.0, centre=[1050.0,1050.0], radius=115.0)

initialise!(model.momentum.p, 0.0)
circle_test3_modified_cells_amount = setField_Circle2D!(mesh=mesh, field=model.momentum.p, value=1.0, centre=[1050.0,1050.0], radius=125.0)

@test circle_test1_modified_cells_amount ≈ circle_test1_expected_cells_amount
@test circle_test2_modified_cells_amount ≈ circle_test2_expected_cells_amount
@test circle_test3_modified_cells_amount ≈ circle_test3_expected_cells_amount



## 3D SPHERE FIELD TEST

# We put the centre of a sphere at (1050, 1050, 100) e.g. offset its centre by 100 from our plane of interest
# Now, let's aim to capture a few nearest cells that intersect with the sphere of a radius 150

#       COORDINATES <> DISTANCE
# 1.    (987.5, 987.5, 0) <> 133.463
# 2.    (962.5, 987.5, 0) <> 146.842
# 3.    (962.5, 987.5, 0) <> 146.842

# If the radius is 100, then only 1 cell at the edge is captured (case #1)
# If the radius is 115, then 3 cells are captured (cases #1 and #3)
# If the radius is 125, then 4 cells are captured (cases #1,#2,#3, while it should not capture #4)


sphere_test_expected_cells_amount = 3

initialise!(model.momentum.p, 0.0)
sphere_test_modified_cells_amount = setField_Sphere3D!(mesh=mesh, field=model.momentum.p, value=1.0, centre=[1050.0,1050.0,100.0], radius=150.0)


@test sphere_test_modified_cells_amount ≈ sphere_test_expected_cells_amount