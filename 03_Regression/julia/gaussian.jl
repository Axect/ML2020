using DataFrames, NCDataFrame, Distributions, LoopVectorization
using LinearAlgebra

function f(x::T) where T <: Number
    sin(x / 10.0) + (x / 50.0)^2
end

function gen_sample()::DataFrame
    normal = Normal(0, 1);
    x = 1.0:100.0;
    ϵ = rand(normal, 100);
    y = Vector{Float64}(undef, 100);
    @avx for i in eachindex(x)
        @inbounds y[i] = f(x[i]) + 0.2 * ϵ[i];      
    end

    DataFrame(x=x, y=y)
end

function ϕ(j::U, s::T, x::T)::T where {U <: Integer, T <: Number}
    if j == 0
        return one(T)
    end
    μ = T(j)
    exp(-(x - μ)^2 / s)
end

function ϕ_vec(s::T, x::T)::Vector{T} where T <: Number
    v = Vector{T}(undef, 101)
    for i in eachindex(v)
        @inbounds v[i] = ϕ(i, s, x)
    end
    v
end

function design_matrix(s::T)::Matrix{T} where T <: Number
    m = Matrix{T}(undef, 100, 101);
    for i ∈ 1:100, j ∈ 1:101
        @inbounds m[i, j] = ϕ(j, s, T(i+1));
    end
    m
end

function w_mle(s::T, t::Vector{T}) where T <: Number
    phi_mat = design_matrix(s)
    pinv(phi_mat) * t 
end

function y(s::T, w::S, x::T)::T where {T <: Number, S <: AbstractArray{T}}
    phi = ϕ_vec(s, x)
    first(w' * phi)
end

function w_ml_reg(s::T, λ::T, t::Vector{T}) where T <: Number
    phi_mat = design_matrix(s)
    λ_eye = λ * I
    pt = phi_mat'
    inv(λ_eye + (pt * phi_mat)) * pt * t
end

function main()
    sample = gen_sample()
    writenc(sample, "data/gauss.nc")
    x = sample[!, :x]
    t = sample[!, :y]

    s = 5.0
    λ = 0.5

    w_reg = w_ml_reg(s, λ, t)
    
    x_draw = 1.0:0.1:100.0;
    y_draw = Vector{Float64}(undef, length(x_draw))
    map!(x -> y(s, w_reg, x), y_draw, x_draw)

    df = DataFrame(x=x_draw, y=y_draw)
    writenc(df, "data/gauss_single_reg.nc")
end