using Random

@kwdef struct Primitives
    b_0::Array{Float64, 1} = [0.5, 1.0]
    x_0::Float64 = 0.0
    T::Int64 = 200
end

mutable struct Data 
    series::Array{Float64, 1} 
    errors::Array{Float64, 2}
end 

function true_DGP(prim::Primitives; rng=Random.default_rng())
    """generate the true data of length T"""

    # unpack parameters
    ρ, σ = prim.b_0

    # generate innovations
    errors = σ .* randn(rng, prim.T)

    # generate AR(1) process
    x = zeros(prim.T)
    x_prev = prim.x_0
    for ix in eachindex(x)
        x[ix] = ρ * x_prev + errors[ix]
        x_prev = x[ix]
    end

    return x 
end

function simulate_errs(H::Int64, prim::Primitives; rng=Random.default_rng())
    """generate a T x H matrix of errors for simulation"""
    return randn(rng, prim.T, H)
end 

function initialize(H::Int64; rng=Random.default_rng())
    """collect data and primitives given H"""

    # initialize primitives 
    prim = Primitives()

    # generate true data 
    series = true_DGP(prim, rng=rng)

    # generate simulate errors 
    errors = simulate_errs(H, prim, rng=rng)

    return Data(series, errors)
end

function simulate(b, prim::Primitives, errors::Matrix{Float64})
    """Generate T × H simulated series given parameters and fixed errors"""

    ρ, σ = b
    T, H = size(errors)

    y = zeros(T, H)

    for h in 1:H
        y_prev = prim.x_0

        for t in 1:T
            y[t, h] = ρ * y_prev + σ * errors[t, h]
            y_prev = y[t, h]
        end
    end

    return y
end

function moments()
    """construct vector of moments"""
end

function smm_objective()
    """compute value of SMM objective given W"""
end

function estimate_smm()
    """two-stage SMM estimation routine"""
end

function newey_west()
    """estimate long-run covariance matrix"""
end

function jacobian()
    """approximate jacobian of moment function"""
end

function ags_statistic()
    """compute AGS statistic"""
end

function j_test()
    """compute value of J-test"""    
end




rng = MersenneTwister(42)
data = initialize(10, rng=rng)