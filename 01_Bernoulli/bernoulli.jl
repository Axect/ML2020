using Pipe

function main(args)
    n = @pipe args[1] |> parse(Int, _)
    d = gen_data(n)
    μ_ml = mle(d)
    println("mle: ", μ_ml)
    bayes = Beta(2, 2)
    @pipe bayes |> update(_, d)
    println(bayes)
    println("optimal: ", bayes |> find_optimal)
end

function gen_data(n)
    rand((0, 1), n)
end

function mle(d::T) where {S <: Number, T <: AbstractArray{S}}
    sum(d) / length(d)
end

mutable struct Beta{T}
    a::T
    b::T
end

function update(β::Beta{T}, d) where T <: Number
    m = count(==(1), d)
    l = length(d) - m
    β.a += m
    β.b += l
end

function find_optimal(β::Beta{T}) where T <: Number
    (β.a - 1) / (β.a + β.b - 2)
end

main(ARGS)
