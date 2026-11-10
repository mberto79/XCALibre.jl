# Assembles changelog/unreleased/*.md into CHANGELOG.md; see changelog/README.md.
#   julia changelog/assemble.jl          preview the next release section
#   julia changelog/assemble.jl v0.7.0   write it into CHANGELOG.md and delete the fragments

using Dates

const CATEGORIES = ["added", "changed", "fixed", "breaking", "deprecated", "removed"]
const ROOT = normpath(joinpath(@__DIR__, ".."))
const FRAGMENTS = joinpath(@__DIR__, "unreleased")
const CHANGELOG = joinpath(ROOT, "CHANGELOG.md")

git(args...) = readchomp(ignorestatus(Cmd(`git -C $ROOT $args`)))

fragment_files() = sort!(filter(f -> endswith(f, ".md") && !startswith(f, "."), readdir(FRAGMENTS)))

function parse_fragment(path)
    entries = Tuple{String,String}[]
    for (i, line) in enumerate(eachline(path))
        isempty(strip(line)) && continue
        m = match(r"^\s*[*-]\s+(\w+):\s*(\S.*)$", line)
        cat = m === nothing ? "" : lowercase(m[1])
        cat in CATEGORIES || error("$(basename(path)):$i: expected `* <category>: text`, with category one of $(join(CATEGORIES, ", "))")
        push!(entries, (cat, rstrip(m[2])))
    end
    entries
end

# Index of the commit on main's first-parent line that brought `file` in, and its PR number.
function landing(file, mainline)
    added = git("log", "--diff-filter=A", "--format=%H", "-1", "--", relpath(file, ROOT))
    isempty(added) && return (typemax(Int), nothing)
    descendants = Set(split(git("rev-list", "--ancestry-path", "$added..HEAD")))
    push!(descendants, added)
    idx = findlast(in(descendants), mainline)
    idx === nothing && return (typemax(Int), nothing)
    subject = git("log", "-1", "--format=%s", mainline[idx])
    m = something(match(r"^Merge pull request #(\d+)", subject), match(r"\(#(\d+)\)\s*$", subject), Some(nothing))
    (length(mainline) - idx + 1, m === nothing ? nothing : m[1])
end

"""
    release_section(heading; fill_links=true)

Return the markdown section built from the unreleased fragments, newest merge first. With
`fill_links`, entries without a `[#N](@ref)` link get the number of the PR that added them.
"""
function release_section(heading; fill_links=true)
    mainline = fill_links ? split(git("rev-list", "--first-parent", "HEAD")) : SubString{String}[]
    files = map(fragment_files()) do f
        path = joinpath(FRAGMENTS, f)
        order, pr = fill_links ? landing(path, mainline) : (0, nothing)
        (; path, order, pr, entries=parse_fragment(path))
    end
    sort!(files, by=f -> -f.order, alg=MergeSort)
    sections = Dict(c => String[] for c in CATEGORIES)
    for f in files, (cat, text) in f.entries
        if fill_links && !occursin(r"\[#\d+\]\(@ref\)", text)
            f.pr === nothing ? @warn("No PR number found for $(basename(f.path)); entry left without a link") :
                (text *= " [#$(f.pr)](@ref)")
        end
        push!(sections[cat], "* " * text)
    end
    io = IOBuffer()
    println(io, "## ", heading, "\n")
    for c in CATEGORIES
        isempty(sections[c]) && continue
        println(io, "### ", uppercasefirst(c))
        foreach(e -> println(io, e), sections[c])
        println(io)
    end
    String(take!(io))
end

# CHANGELOG.md with `section` inserted before the latest released version.
function with_section(section)
    text = read(CHANGELOG, String)
    i = findfirst(r"^## "m, text)
    i === nothing ? text * "\n" * section : text[1:first(i)-1] * section * "\n" * text[first(i):end]
end

function main(args)
    if isempty(args)
        print(release_section("Unreleased"))
        return
    end
    version = only(args)
    occursin(r"^v\d+\.\d+\.\d+$", version) || error("version must look like v0.7.0, got $version")
    occursin("[$version]", read(CHANGELOG, String)) && error("CHANGELOG.md already has $version")
    isempty(fragment_files()) && error("no fragments in changelog/unreleased")
    write(CHANGELOG, with_section(release_section("Version [$version] - $(today())")))
    foreach(f -> rm(joinpath(FRAGMENTS, f)), fragment_files())
    println("Wrote $version to CHANGELOG.md and removed the fragments; review and commit.")
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
