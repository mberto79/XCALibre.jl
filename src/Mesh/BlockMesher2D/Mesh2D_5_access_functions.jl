export x, y, z # access cell centres
export xf, yf, zf # access face centres


function x(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[1]
    end
    return out
end

function y(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[2]
    end
    return out
end

function z(mesh::Mesh2{I,F}) where {I,F}
    cells = mesh.cells
    out = zeros(F, length(cells))
    @inbounds for i ∈ eachindex(cells)
        out[i] = cells[i].centre[3]
    end
    return out
end

function xf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[1]
    end
    return out
end

function yf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[2]
    end
    return out
end

function zf(mesh::Mesh2{I,F}) where {I,F}
    faces = mesh.faces
    out = zeros(F, length(faces))
    @inbounds for i ∈ eachindex(faces)
        out[i] = faces[i].centre[3]
    end
    return out
end