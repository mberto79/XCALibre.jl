# Kernels must bind @index(...) on its own (`i = @index(Global)`): an @index nested in an
# expression (`@index(Global) + n`) or assigned into an array or field compiles on the CPU
# only with KernelAbstractions 0.9.45 or later, below the version range the package allows.
let root = pkgdir(XCALibre)
    standalone = r"^\s*[A-Za-z_]\w*(\s*,\s*[A-Za-z_]\w*)*\s*=\s*@index\([^()]*\)\s*(#.*)?$"
    offending = String[]
    for dir in ("src", "ext"), (path, _, files) in walkdir(joinpath(root, dir)), f in files
        endswith(f, ".jl") || continue
        for (n, line) in enumerate(eachline(joinpath(path, f)))
            code = strip(line)
            (startswith(code, "#") || !occursin("@index(", code)) && continue
            occursin(standalone, line) || push!(offending, "$(relpath(joinpath(path, f), root)):$n: $code")
        end
    end
    isempty(offending) || @info "Nested @index uses" offending
    @test isempty(offending)
end
