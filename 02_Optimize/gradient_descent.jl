using ForwardDiff: jacobian

function main()
    T = Float64[1,2,3,4,5]
    Y = Float64[0.8, 2.1, 3, 4.1, 5]
    Yh = p -> @. p[1] * T + p[2]
    P_init = Float64[2,-1]
    println(Y)
    println(Yh(P_init))
    println(jacobian(Yh, P_init))

    opt = Optimizer(Y, Yh, P_init)
    P = opt |> optimize
    println(P)
end

# Optimizer
mutable struct Optimizer{T}
    Y::Vector{T}
    Yh::Function
    P::Vector{T}
    α::T
    num::Int
end

# Constructor
function Optimizer(Y, Yh, P, α=1e-2, num=10000)
    new(Optimizer(Y, Yh, P, α, num))
end

function set_step(opt::Optimizer{T}, α) where T <: Number
    opt.α = α
end

function set_iteration(opt::Optimizer{T}, num) where T <: Number
    opt.num = num
end

function get_sse(opt::Optimizer)
    error = opt.Y - opt.Yh(opt.P)
    (error' * error)[1,1]
end

function get_param(opt::Optimizer)
    opt.P
end

function update(opt::Optimizer)
    J = jacobian(opt.Yh, opt.P)
    h = 2 * opt.α * J' * (opt.Y - opt.Yh(opt.P))
    opt.P = opt.P .+ h
end

function optimize(opt::Optimizer)
    for i in 1:opt.num
        opt |> update
        opt |> get_sse |> println
    end
    opt |> get_param
end

main()

