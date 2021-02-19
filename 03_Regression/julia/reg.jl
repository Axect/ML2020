using NCDataFrame, Distributions, DataFrames, LinearAlgebra

function main()
    df = gen_sample(100)

    x = df[!,"x"]
    t = df[!,"t"]

    println(df)

    writenc(df, "data/data.nc")

    ϕ_mat = ϕ(x)
    m = m_n(ϕ_mat, t)

    x_plot = -1:0.1:1
    y_plot = map(x -> m[1] + m[2]*x, x_plot)

    dg = DataFrame()
    dg[!,"x"] = x_plot
    dg[!,"y"] = y_plot

    println(dg)

    writenc(dg, "data/reg.nc")
end

function f(x)
    -0.3 .+ 0.5 * x
end

function gen_sample(n)
    x = rand(Uniform(-1, 1), 100)
    n = Normal(0, 0.2)
    ϵ = rand(n, 100)
    t = f(x) .+ ϵ

    df = DataFrame(x=x, t=t)
end

function ϕ(x::Vector{T}) where T <: Number
    n = length(x)
    ph = ones(n, 2)
    ph[:, 2] = x
    ph
end

function s_n_inv(ph::T)::T where T <: AbstractMatrix
    2 * I + 25 * (ph' * ph)
end

function m_n(ph::Matrix{T}, t::Vector{T}) where T <: Number
    s = s_n_inv(ph) |> inv
    s * ph' * (t * 25)
end

main()
