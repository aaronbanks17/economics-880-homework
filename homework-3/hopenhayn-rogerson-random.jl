@kwdef struct Primitives

    # parameters
    β::Float64 = 0.8       # discount factor
    θ::Float64 = 0.64      # production curvature
    c_f::Float64 = 10.0    # fixed operating cost
    c_e::Float64 = 5.0     # entry cost
    A::Float64 = 0.005     # labor disutility
    α::Float64 = 1.0       # TV1 shock precision

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
    val_func::Vector{Float64}     # U(s; p)
    exit_pol::Vector{Float64}     # probability of exit
    labor_pol::Vector{Float64}    # n_star(s; p)
    profits::Vector{Float64}      # π(s; p)

    # stationary distribution
    sta_dist::Vector{Float64}     # μ(s), incumbent firm masses

    # equilibrium objects
    p::Float64                    # output price
    M::Float64                    # mass of entrants

end


function initialize(; α::Float64 = 1.0)

    # initialize primitives
    prim = Primitives(α = α)

    # firm-level objects
    val_func = zeros(prim.n_s)
    exit_pol = zeros(prim.n_s)
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


function entry_value(res::Results, prim::Primitives)
    """calculate expected value of entry given p"""

    return sum(res.val_func .* prim.ν) - res.p * prim.c_e
end

# Euler-Mascheroni constant
const γ_E = 0.5772156649015329

# numerically stable log(1 + exp(x))
softplus(x::Float64) = max(0.0, x) + log1p(exp(-abs(x)))


function bellman(res::Results, prim::Primitives)
    """Apply TV1 Bellman operator to current value-function guess."""

    (; val_func, profits) = res
    (; β, F, α) = prim

    # expected continuation values
    EV = F * val_func

    # scaled continuation values
    z = α * β .* EV

    # updated ex-ante value function
    v_next = profits .+ γ_E / α .+ softplus.(z) ./ α

    return v_next
end


function solve_firm!(
    res::Results,
    prim::Primitives;
    tol::Float64 = 1e-10,
    max_iter::Int64 = 10_000
)
    """Solve TV1 firm problem at current price."""

    (; s_grid, F, β, θ, c_f, α) = prim
    (; p) = res

    # static labor demand
    res.labor_pol .= (p * θ .* s_grid) .^ (1 / (1 - θ))

    # operating profits
    res.profits .= (
        p .* s_grid .* (res.labor_pol .^ θ)
        .- res.labor_pol
        .- p * c_f
    )

    # value-function iteration
    for iter in 1:max_iter

        # apply Bellman operator
        v_next = bellman(res, prim)

        # convergence error
        err = maximum(abs.(v_next .- res.val_func))

        # update value function
        res.val_func .= v_next

        if err < tol

            # scaled expected continuation values
            z = α * β .* (F * res.val_func)

            # exit probabilities
            res.exit_pol .= exp.(-softplus.(z))

            return res.val_func, res.exit_pol, res.labor_pol
        end
    end

    error("Firm value-function iteration did not converge.")
end

function solve_prices!(
    res::Results,
    prim::Primitives;
    p0::Float64 = 1.0,
    tol::Float64 = 1e-8
)
    """Find equilibrium output price satisfying free entry."""

    # ensure positive initial price
    p0 > 0 || error("Initial price must be positive.")

    # free-entry residual as a function of log price
    function excess_entry(log_p)

        # recover positive candidate price
        res.p = exp(log_p)

        # solve firm's dynamic programming problem
        solve_firm!(res, prim)

        # return normalized free-entry residual
        return entry_value(res, prim) / res.p
    end

    # find root using secant method
    log_p_star = find_zero(
        excess_entry,
        log(p0),
        Order1();
        atol = tol
    )

    # store equilibrium price
    res.p = exp(log_p_star)

    # ensure policies correspond to equilibrium price
    solve_firm!(res, prim)

    # verify free-entry condition
    err = abs(entry_value(res, prim) / res.p)

    if err > tol
        error("Free-entry condition not satisfied: residual = $err")
    end

    return res.p
end

function stationary_distribution(
    res::Results,
    prim::Primitives
)
    """Compute stationary incumbent distribution per unit of entry."""

    (; F, ν, n_s) = prim
    (; exit_pol) = res

    # survival probabilities
    survive = 1.0 .- exit_pol

    # transition matrix including survival
    T = transpose(F) * Diagonal(survive)

    # stationary distribution for M = 1
    μ_unit = (Matrix{Float64}(I, n_s, n_s) - T) \ (T * ν)

    return μ_unit
end


function solve_labor_market!(
    res::Results,
    prim::Primitives
)
    """Solve for equilibrium entrant mass and firm distribution."""

    (; ν, A) = prim
    (; labor_pol, profits) = res

    # stationary incumbent distribution per unit of entry
    μ_unit = stationary_distribution(res, prim)

    # total operating firms per unit of entry
    firms_unit = μ_unit .+ ν

    # aggregate labor demand per unit of entry
    L_unit = sum(firms_unit .* labor_pol)

    # aggregate profits per unit of entry
    Π_unit = sum(firms_unit .* profits)

    # equilibrium mass of entrants
    M_star = 1.0 / (A * (L_unit + Π_unit))

    # update equilibrium quantities
    res.M = M_star
    res.sta_dist .= M_star .* μ_unit

    return nothing
end