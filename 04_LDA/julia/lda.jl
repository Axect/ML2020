using DataFrames, NCDataFrame
using LoopVectorization
using LinearAlgebra, Distributions, Statistics

"""
    Weight

...
## Description
- x: 300 x 3 (Matrix)
- t: 300 x 2 (Matrix)
- w:   3 x 2 (Matrix)
...
"""
function weight_ls(x::S, t::S) where {T <: Number, S <: AbstractMatrix{T}}
    pinv(x) * t
end

"""
    Least square fitting (single)

...
## Description
- w: 3 x 2 (Matrix)
- x: 2 x 1 (Vector)
- z: 3 x 1 (Vector)
...
"""
function least_square(w::S, x::V) where {T <: Number, S <: AbstractMatrix{T}, V <: AbstractVector{T}}
    z = vcat([one(eltype(x))], x)
    w' * z
end

"""
    Least square fitting (vectorized)

...
## Description
- n : size of domain
- w : 3 x 2 (Matrix)
- x : 2 x n (Matrix)
- z : 3 x n (Matrix)
- y : 2 x n (Matrix)
...
"""
function least_square_map(w::S, x::S) where {T <: Number, S <: AbstractMatrix{T}}
    z = S(undef, 3, size(x)[2])
    z[1,:] .= ones(eltype(x), size(x)[2])
    z[2,:] .= x[1,:]
    z[3,:] .= x[2,:]

    w' * z
end

"""
    Classifier
"""
function classifier(y::S) where {T <: Number, S <: AbstractVector{T}}
    argmax(y) + 1
end

"""
    Decision Boundary 1
"""
function boundary_1(w::S, x::V) where {T <: Number, S <: AbstractMatrix{T}, V <: AbstractVector{T}}
    z = V(undef, length(x))
    @simd for i in eachindex(x)
        @inbounds z[i] = (x[i] * (-w[2,1]) - w[1,1] + 0.5) / w[3,1]
    end
    z
end

"""
    Decision Boundary 2
"""
function boundary_2(w::S, x::V) where {T <: Number, S <: AbstractMatrix{T}, V <: AbstractVector{T}}
    z = V(undef, length(x))
    @simd for i in eachindex(z)
        @inbounds z[i] = (x[i] * (-w[2,2]) - w[1,2] + 0.5) / w[3,2]
    end
    z
end

"""
    Fisher's LDA
"""
function weight_fisher(s_w::S, m1::V, m2::V) where {T <: Number, S <: AbstractMatrix{T}, V <: AbstractVector{T}}
    inv(s_w) * (m2 - m1)
end

"""
    Boundary of Fisher's LDA
"""
function boundary_fisher(w::V, x::V, m::V) where {T <: Number, V <: AbstractVector{T}}
    (x .- m[1]) * (-w[1] / w[2]) .+ m[2]
end

"""
    Main function

...
## Description
- x : 300 x 1 (Vector)
- y : 300 x 1 (Vector)
- z : 300 x 3 (Matrix) - Input Data
- t1 : 300 x 1 (Vector)
- t2 : 300 x 1 (Vector)
- t : 300 x 2 (Matrix) - One hot encoding
...
"""
function main()
    # Generate 2D Data
    n1 = Normal(3, 1)
    n2 = Normal(1, 3)
    n3 = Normal(-3, 1)
    n4 = Normal(-1, 3)

    x1 = rand(n1, 150)
    y1 = rand(n2, 150)
    x2 = rand(n3, 150)
    y2 = rand(n4, 150)

    # Data for Least square
    x = vcat(x1, x2)
    y = vcat(y1, y2)
    z = hcat(ones(eltype(x), 300), x, y)
    t1 = vcat(ones(eltype(x), 150), zeros(eltype(x), 150))
    t2 = vcat(zeros(eltype(x), 150), ones(eltype(x), 150))
    t = hcat(t1, t2)

    # Weight computation
    # w : 3 x 2 (Matrix)
    w = weight_ls(z, t)
    @show w

    # Test Classification
    l1 = least_square(w, Float64[3, 1])
    @show l1
    l2 = least_square(w, Float64[-3, 1])
    @show l2
    @show classifier(l2)

    # To draw decision boundary
    domain = collect(-5.0:0.01:5.0)
    b1 = boundary_1(w, domain)
    b2 = boundary_2(w, domain)

    # Fisher
    g1 = hcat(x1, y1)
    g2 = hcat(x2, y2)
    m1 = vec(mean(g1, dims=1))
    m2 = vec(mean(g2, dims=1))
    
    @fastmath m = (m1 .+ m2) ./ 0.5;
    @show m

    s = cov(g1) .+ cov(g2)
    @show s
    @show inv(s)

    w_fisher = weight_fisher(s, m1, m2)
    @show w_fisher
    normalize!(w_fisher)
    @show w_fisher

    w_g1 = g1 * w_fisher
    w_g2 = g2 * w_fisher
    w_0 = dot(m, w_fisher)

    max_l = length(domain)

    df = DataFrame(
        x1=fillmissing(x1, max_l),
        y1=fillmissing(y1, max_l),
        x2=fillmissing(x2, max_l),
        y2=fillmissing(y2, max_l),
        d=fillmissing(domain, max_l),
        b1=fillmissing(b1, max_l),
        b2=fillmissing(b2, max_l),
        bf=fillmissing(boundary_fisher(w_fisher,domain,m), max_l),
        r1=fillmissing(w_g1, max_l),
        r2=fillmissing(w_g2, max_l),
    )

    writenc(df, "data/lda.nc")
end
