struct Linear end
struct Limited end
struct Constant end

struct Laplacian{T}
    J::Float64
    phi::Vector{Float64}
    sign::Vector{Int64}
end

struct VariableLaplacian{T}
    J::Vector{Float64}
    phi::Vector{Float64}
    sign::Vector{Int64}
end

struct Source{T}
    phi::Vector{Float64}
    sign::Vector{Int64}
end

cells = Int(10) 
phi = ones(cells)
phiSource =  zeros(cells)
J1 = 0.5
J2 = Float64[1.0,2.0,3.0,4.0]
sign = [1]

term1 = Laplacian{Linear}(J1, phi,[1])
term2 = Laplacian{Limited}(J1, phi, [1])
# term2 = VariableLaplacian{Limited}(J2, phi, [1])
src1 = Source{Constant}(phiSource, [1])

macro discretise_macro(ex)
    model = esc(ex)
    name = esc(:discretise_fn)
    A = esc(:A)

    insert = Expr(:block)
    for term ∈ [:(model.term1), :(model.term2)]
        push!(insert.args, :(A[i] += $(term).J) )
    end

    forLoop = :(for i ∈ eachindex(A); $insert; end)
    quote
    return function $name($A, $model::Model0)
        $(esc(forLoop))
    end
    end
end

t = @macroexpand(@discretise_macro model)
discretise_macro! = @discretise_macro model
@time discretise_macro!(A, model)
A
A = Float64[1:cells;]
A


macro model_builder_tuple(modelName::String, terms::Integer, sources::Integer)
    name = Symbol(modelName)

    param_types = []
    fieldsnames = []
    
    terms_tuple = Expr(:tuple)
    constructor_terms = Expr(:tuple)
    terms_types = Expr(:curly, :Tuple)
    for t ∈ 1:(terms)
        tt = Symbol("T$t") |> esc
        ts = Symbol("term$t")
        push!(param_types, tt)
        push!(terms_types.args, tt)
        push!(fieldsnames, ts)
        push!(constructor_terms.args, :($ts=$ts))
        push!(terms_tuple.args, QuoteNode(ts))
        # push!(fields, Expr(:(::), Symbol("term$t"), tt))
        
    end
    
    sources_tuple = Expr(:tuple)
    constructor_sources = Expr(:tuple)
    sources_types = Expr(:curly, :Tuple)
    for s ∈ 1:(sources)
        ss = Symbol("S$s") |> esc
        src_symbol = Symbol("source$s")
        push!(param_types, ss)
        push!(sources_types.args, ss)
        push!(fieldsnames, src_symbol)
        push!(constructor_sources.args, :($src_symbol=$src_symbol))
        push!(sources_tuple.args, QuoteNode(src_symbol))
        # push!(fields, Expr(:(::), Symbol("src$s"), ss))
    end

    terms = Expr(:(::), :terms, :(NamedTuple{$terms_tuple, $terms_types}))
    sources = Expr(:(::), :sources, :(NamedTuple{$sources_tuple, $sources_types}))

    structBody = Expr(:block, Expr(
        :struct, false, Expr(:curly, name, param_types...),
        Expr(:block, terms, sources)
        ))

    func = :(function $(name)($(fieldsnames...))
        begin
        $(name)(($(constructor_terms)),$(constructor_sources))
        end
    end) |> esc
    out = quote begin
        $structBody
        $func
    end end
    return  out #structBody, func
end

@macroexpand(@model_builder_tuple "SteadyDiffusion" 1 1)

@model_builder_tuple "SteadyDiffusion" 2 1

model1 =SteadyDiffusion(term1, term1, src1)

macro discretise(eqn_expr)
    eqn = esc(eqn_expr)
    # terms = :((term1))
    terms = [:(term1)]

    # terms = $(eqn).terms
    # collect_expr = Symbol[]
    # for term ∈ terms
    #     push!(collect_expr, term.arg)
    # end
    # args = Expr(:tuple, collect_expr...)
    
    

    notBoundaryBranch = Base.remove_linenums!(:(if c1 != c2 end))

    # aP = :(A[cID, cID] += Γ*face.area*norm(face.normal)/face.delta)
    # aN = :(A[cID, neighbour] += -Γ*face.area*norm(face.normal)/face.delta)

    for term ∈ terms
        push!(
            func.args[2].args[2].args[2].args[2].args[2].args,
            # term.aP
            :(aP!(A, $term, face, cID))
            )

        push!(
            notBoundaryBranch.args[2].args,
            # term.aN
            :(aN!(A, $term, face, cID, nID))
            )
    end
    push!(
        func.args[2].args[2].args[2].args[2].args[2].args,
        notBoundaryBranch)
    quote
        function discretise!(model, eqn)
            begin
                mesh = model.terms.term1.ϕ.mesh
                cells = mesh.cells
                faces = mesh.faces
                nCells = length(cells)
                A = eqn.A
            end
            for cID ∈ 1:nCells
                cell = cells[cID]
                for fi ∈ eachindex(cell.facesID)
                    fID = cell.facesID[fi]
                    face = faces[fID]
                    nID = cell.neighbours[fi]
                    c1 = face.ownerCells[1]
                    c2 = face.ownerCells[2]
                    # discretisation code here
                end
            end
            nothing
        end # end Function
    end  # end quote
    end
