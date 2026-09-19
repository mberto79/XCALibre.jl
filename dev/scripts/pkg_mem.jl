# Private and shared MB added by loading one package in a fresh process; see dev/scripts/INDEX.md.
# `one <Name>...` loads the named packages (transitive deps resolved through the manifest); `list` prints what a worker loads.
using TOML

function smaps()
    d = Dict{String,Float64}()
    for l ∈ eachline("/proc/self/smaps_rollup")
        f = split(l)
        length(f) == 3 && f[3] == "kB" && (d[chop(f[1])] = parse(Int, f[2]) / 1024)
    end
    (priv=d["Private_Clean"] + d["Private_Dirty"], shared=d["Shared_Clean"] + d["Shared_Dirty"], pss=d["Pss"])
end

function pkgid(name)
    m = TOML.parsefile(joinpath(dirname(Base.active_project()), "Manifest.toml"))
    Base.PkgId(Base.UUID(m["deps"][name][1]["uuid"]), name)
end

if ARGS[1] == "one"
    ids = pkgid.(ARGS[2:end])
    a = smaps()
    t = @elapsed foreach(Base.require, ids)
    b = smaps()
    println("PKG ", join(ARGS[2:end], "+"), " priv_MB=", round(b.priv - a.priv, digits=1), " shared_MB=", round(b.shared - a.shared, digits=1),
        " load_s=", round(t, digits=2), " base_priv_MB=", round(a.priv, digits=1), " modules=", length(Base.loaded_modules))
elseif ARGS[1] == "list"
    using XCALibre, PETSc, MPI, Libdl
    println("MODULES ", length(Base.loaded_modules), " ", join(sort([k.name for k ∈ keys(Base.loaded_modules)]), " "))
    println("LIBS ", join(sort(filter(l -> occursin(r"^lib(petsc|mpi|LLVM|cuda|hypre)"i, basename(l)), Libdl.dllist())), "\n"))
end
