using XCALibre
using InteractiveUtils
const FM = XCALibre.FoamMesh

# ---------- 1. LINE-BREAK ROBUSTNESS TEST (user-required) ----------
mktempdir() do dir
    faces_single = """FoamFile
{
    class faceList;
}
3
(
4(0 1 2 3)
3(4 5 6)
4(7 8 9 10)
)
"""
    faces_multi = """FoamFile
{
    class faceList;
}
3
(
4
(
0 1 2 3
)
3
(
4 5 6
)
4
(
7 8 9 10
)
)
"""
    pts_single = """FoamFile
{ class pointField; }
2
(
(0.0 1.5 -2.0)
(3.0 -4.5 6.0)
)
"""
    pts_multi = """FoamFile
{ class pointField; }
2
(
(
0.0 1.5 -2.0
)
(
3.0 -4.5 6.0
)
)
"""
    write(joinpath(dir,"faces_s"), faces_single); write(joinpath(dir,"faces_m"), faces_multi)
    write(joinpath(dir,"pts_s"), pts_single);     write(joinpath(dir,"pts_m"), pts_multi)

    fs = FM.read_faces(joinpath(dir,"faces_s"), Int64, Float64)
    fm = FM.read_faces(joinpath(dir,"faces_m"), Int64, Float64)
    ps = FM.read_points(joinpath(dir,"pts_s"), 1.0, Int64, Float64)
    pm = FM.read_points(joinpath(dir,"pts_m"), 1.0, Int64, Float64)

    println("LINEBREAK faces single == multi : ", fs == fm, "  ", fs)
    println("LINEBREAK points single == multi: ", ps == pm, "  ", ps)
    println("LINEBREAK OVERALL: ", (fs == fm && ps == pm) ? "PASS" : "FAIL")
end

# ---------- 2. generate_faces type-stability diagnosis ----------
dir = "test/grids/OF_cavity_hex/polyMesh"
fd = FM.read_FOAM3D(dir, 1.0, Int64, Float64)
conn = FM.connect_mesh(fd, Int64, Float64)
io = IOBuffer()
code_warntype(io, FM.generate_faces, (typeof(fd), typeof(conn), Type{Int64}, Type{Float64}))
s = String(take!(io))
println("==== code_warntype generate_faces ====")
println(s)
println("==== ANY/UNION line count: ", count(l->occursin("::Any", l)||occursin("Union{", l), split(s,'\n')))
