@kwdef struct Primitives

    # parameters
    β::Float64 = 0.8       # discount factor
    θ::Float64 = 0.64      # production curvature
    c_f::Float64 = 10.0    # fixed operating cost
    c_e::Float64 = 5.0     # entry cost
    A::Float64 = 0.005     # labor disutility

    # productivity states
    s_grid::Vector{Float64} = [
        3.98e-4, 3.58, 6.82, 12.18, 18.79
    ]
    n_s::Int64 = length(s_grid)

    # productivity transition matrix
    F::Matrix{Float64} = [
        0.6598  0.2600  0.0416  0.0331  0.0055;
        0.1997  0.7201  0.0420  0.0326  0.0056;
        0.2000  0.2000  0.5555  0.0344  0.0101;
        0.2000  0.2000  0.2502  0.3397  0.0101;
        0.2000  0.2000  0.2500  0.3400  0.0100
    ]

    # entrant productivity distribution
    ν::Vector{Float64} = [
        0.37, 0.4631, 0.1102, 0.0504, 0.0063
    ]

end

mutable struct Results

    # firm-level objects
    val_func::Vector{Float64}     # W(s; p)
    exit_pol::Vector{Bool}        # X(s; p), true = exit
    labor_pol::Vector{Float64}    # n*(s; p)
    profits::Vector{Float64}      # π(s; p)

    # stationary distribution
    sta_dist::Vector{Float64}     # μ(s), incumbent firm masses

    # equilibrium objects
    p::Float64                    # output price
    M::Float64                    # mass of entrants

end

function initialize()

    # initialize primitives
    prim = Primitives()

    # firm-level objects
    val_func = zeros(prim.n_s)
    exit_pol = falses(prim.n_s)
    labor_pol = zeros(prim.n_s)
    profits = zeros(prim.n_s)

    # stationary incumbent distribution
    sta_dist = zeros(prim.n_s)

    # initial aggregate guesses
    p = 1.0
    M = 1.0

    # package endogenous objects
    res = Results(
        val_func,
        exit_pol,
        labor_pol,
        profits,
        sta_dist,
        p,
        M
    )

    return prim, res

end

function labor_demand(res::Results, prim::Primitives)
    """calculate firms' labor demadn given (p, s)"""

end

function profits(res::Results, prim::Primitives)
    """calculate firm profits given (p, s)"""
end

function bellman(res::Results, prim::Primitives)
    """
    bellman operate to update guess of operating
    firms value function and exit decision
    """
    
end

function entry_value(res::Results, prim::Primitives)
    """calculate expected value of entry given p"""

end

function solve_prices()
    """find equilibrium price for given mass of firms"""
    
end