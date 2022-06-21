struct Test0{T}
    value::T
end

struct Test1{T}
    value::T
end

struct Test2{T}
    value::T
end

a = Test0(1.0)
b = Test1(2)
c = Test2(3.5)

BC = (a, b, c)

t = typeof(BC)

times2(a) = 2*a.value

@generated function apply2(var, BC::Tuple)
    calls = Expr[]
    for i ∈ 1:length(BC.parameters)
        func_call = :(sum += times2(BC[$i]))
        push!(calls, func_call)
    end

    quote
        println("You called var = ", var)
        println("Make sum = 0")
        sum = 0.0
        $(calls...)
        return sum
    end
end

@code_warntype apply2(30, BC)

tt = quote
    for i ∈ 1:20
        a += i*2
    end
end

vt = Expr[tt, tt, tt]

vt

exs = Expr(:block, vt...)
qt = quote
    $(vt...)
end
calls = Vector{Expr}()
