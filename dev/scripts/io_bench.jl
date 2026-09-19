# Decomposed-writer timing: mesh plus U, p (and phi for binary) written by the current binary writer and by
# the ASCII writer at 590b30e3, loaded from git under renamed names; prints the slowest rank's seconds and MB.
using XCALibre, PETSc, MPI
const D = XCALibre.Distribute
dir, out = ARGS[1], ARGS[2]

old = read(`git -C $(pkgdir(XCALibre)) show 590b30e3:src/Distribute/Distribute_7_io.jl`, String)
old = old[findfirst("# NEW SECTION: decomposed OpenFOAM writer", old)[1]:findfirst("# NEW SECTION: writer dispatch", old)[1]-1] *
    old[findfirst("# NEW SECTION: field output", old)[1]:end]
for (a, b) ∈ ("PFOAMWriter" => "AsciiWriter", "initialise_writer(format::OpenFOAM" => "ascii_initialise_writer(format::OpenFOAM",
        "function write_results(" => "function ascii_write_results(", "_foam_header(" => "_ascii_header(",
        "_proc_patch_value(" => "_ascii_ppv(", "_pface_layout(" => "_ascii_layout(", "_pface_owner_flip(" => "_ascii_flip(",
        "_write_face_nodes(" => "_ascii_face_nodes(")
    global old = replace(old, a => b)
end
include_string(D, old)

MPI.Init()
comm = MPI.COMM_WORLD
dm = distribute(dir; comm)
bcs = assign(region=dm, (
    U = [Dirichlet(:inlet, [0.5, 0.0, 0.0]), Zerogradient(:outlet), Wall(:wall, [0.0, 0.0, 0.0]),
         Zerogradient(:sides), Zerogradient(:top)],
    p = [Zerogradient(:inlet), Dirichlet(:outlet, 0.0), Wall(:wall), Extrapolated(:sides), Extrapolated(:top)]))
U, p, phi = VectorField(dm), ScalarField(dm), FaceScalarField(dm)
foreach(v -> v .= rand(length(v)), (U.x.values, U.y.values, U.z.values, p.values, phi.values))
mkpath(out); cd(out)
fields = (("U", U), ("p", p))
function timed(f)
    f()
    MPI.Barrier(comm)
    t = @elapsed (f(); MPI.Barrier(comm))
    t
end
tb = timed(() -> (w = initialise_writer(OpenFOAM(), dm); attach_flux!(w, phi); write_results(1, 1, dm, w, bcs, fields...)))
mb_b = sum(filesize, [joinpath(r, f) for (r, _, fs) ∈ walkdir("processor$(MPI.Comm_rank(comm))") for f ∈ fs]) / 2^20
rm("processor$(MPI.Comm_rank(comm))"; recursive=true)
ta = timed(() -> (w = D.ascii_initialise_writer(OpenFOAM(), dm); D.ascii_write_results(1, 1, dm, w, bcs, fields...)))
mb_a = sum(filesize, [joinpath(r, f) for (r, _, fs) ∈ walkdir("processor$(MPI.Comm_rank(comm))") for f ∈ fs]) / 2^20
MPI.Comm_rank(comm) == 0 && println("IOBENCH n=$(MPI.Comm_size(comm)) binary_s=$(round(tb; digits=2)) ",
    "binary_MB_rank0=$(round(mb_b; digits=1)) ascii_s=$(round(ta; digits=2)) ascii_MB_rank0=$(round(mb_a; digits=1)) (binary includes phi)")
