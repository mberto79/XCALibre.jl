# struct Upwind end
# struct LinearUpwind end

# struct Term0{T}
#     a::Int64
#     b::Float64
#     type::T
# end

# struct Term1{T}
#     a::Int64
#     b::Float64
#     c::Float64
#     type::T
# end
# t0 = Term0{Upwind}(1,1.0,Upwind())
# t1 = Term1{LinearUpwind}(4, 1.0, 2.0,LinearUpwind())

# # terms = [t0, t1]

# # Tterms = (terms...,)

# # isbits(Tterms)

# # typeof(Tterms)

# # struct TestIsBits
# #     terms
# # end

# # struct TestIsBits1
# #     terms::Tuple{FVM_1D.Term0{FVM_1D.Upwind}, FVM_1D.Term1{FVM_1D.LinearUpwind}}
# # end

# # struct TestIsBits2{T1,T2}
# #     terms::Tuple{T1, T2}
# # end

# # tt = TestIsBits2(Tterms)

# # isbits(tt)

# # macro terms(ex)
# #     terms = esc(ex)
# #     quote
# #         nTerms = length($terms)
# #         types = [Symbol("T$i") for i ∈ 1:nTerms]
# #         # T = :({$(types...)})
# #         braces = Expr(:braces, types...)
# #         structDef = Expr(:curly, :Ter, types...)
# #         typeDef = Expr(:curly, :Tuple, types...)

# #         t = quote
# #             struct $structDef
# #                 terms::$typeDef
# #             end
# #         end 
# #         eval(t)
# #     end
# # end

# # ts = @terms terms

# # [Symbol("T$t") for t ∈ 1:5]
# # Expr(:curly, :term, [i for i ∈ [[Symbol("T$t") for t ∈ 1:5]...]]...)
# # i = 2
# # ttt = quote {T1,T2} end
# # dump(ttt)

# # struct Lin end
# # struct Operation0{T}
# #     Γ
# #     ϕ
# #     discretisation::T
# #     operator::Symbol
# # end
# # Operation0{Linear}(Γ, ϕ) = Operation0(Γ, ϕ, Lin(), :laplacian)

# # Op = Operation0{Linear}(1,2)

# # function build_term(op::Operation)
# #     op.operator

# tup = (t0, t1)

# struct Eqn6{T1,T2,S1,S2}
#     terms::Tuple{T1,T2}
#     nTerms::Int64
#     tSign::Vector{Int64}
#     sources::Tuple{S1,S2}
#     nSources::Int64
#     tSources::Vector{Int64}
# end

# v1 = 1
# v2 = 2
# e = Eqn6{Term0{Upwind}, Term1{LinearUpwind}, Int64, Int64}((t0, t1), 2, [1,1], ([v1, v2]...,), 3, [1,4,6])
# isbits(e)
# typeof(e)
# q = ([t0, t1]...,)
# isbits(q)
# typeof(q)

# function test(qq)
#     # for qi ∈ q
#     for i ∈ qq
#     bb = i.a
#     aa = i.a
#     end
#     # end
# end
# @code_warntype test(q)

# function fn_loop(f, vals)
#     for fi ∈ f
#         fi.(vals)
#     end
# end

# fn = (sin, cos, tan)
# vals = [1,2,3]
# function test(fn , vals)
#     fn_loop(fn,vals)
# end
# @code_warntype fn_loop(fn,vals)
# @time fn_loop(fn,vals)
