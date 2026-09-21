using Parameters, LinearAlgebra, Printf

# ═══════════════════════════════════════════════════════════════
# 1. MODEL PARAMETERS
# ═══════════════════════════════════════════════════════════════

@with_kw struct Primitives
    β::Float64 = 0.9932                  # discount factor
    α::Float64 = 1.5                     # CRRA coefficient of relative risk aversion

    # Earnings process
    S::Vector{Float64} = [1.0, 0.5]      # earnings levels: S[1] = e (employed), S[2] = u (unemployed)
    ns::Int64 = length(S)
    Π::Matrix{Float64} = [0.97 0.03;     # Π[i, j] = Prob(s' = S[j] | s = S[i])
                          0.50 0.50]     

    # Asset grid (discretized A = [a_min, a_max])
    a_min::Float64 = -2.0                # borrowing limit
    a_max::Float64 = 5.0                 # upper bound;
    na::Int64 = 1000                     # grid points
    a_grid::Vector{Float64} = collect(range(a_min, a_max, length = na))
end


# ═══════════════════════════════════════════════════════════════
# 2. RESULTS
# ═══════════════════════════════════════════════════════════════

mutable struct Results
    v::Matrix{Float64}          # value function v(a, s; q)
    pol_idx::Matrix{Int64}      # ι*(a, s): grid INDEX of optimal a' — what T* needs
    pol::Matrix{Float64}        # g(a, s) = a_grid[pol_idx]
    μ::Matrix{Float64}          # invariant distribution over (a, s); sums to 1
    q::Float64                  # current bond price guess (updated by the outer loop)
end

# ═══════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# CRRA period utility; -Inf rules out c ≤ 0 inside the max
u(c::Float64, α::Float64) = c > 0.0 ? (c^(1 - α) - 1) / (1 - α) : -Inf


# Stationary distribution π* of the earnings chain
function stationary_earnings(prim::Primitives)
    @unpack Π = prim
    eig = eigen(Matrix(Π'))
    idx = argmin(abs.(eig.values .- 1.0))    # eigenvalue 1 ↔ stationary vector
    π_star = real.(eig.vectors[:, idx])
    return π_star ./ sum(π_star)
end


# Build primitives and a starting point for results
function Initialize(; q0::Float64 = 0.9966)  # q0 = midpoint of (β, 1]
    prim = Primitives()
    @unpack na, ns = prim

    v = zeros(na, ns)                         
    pol_idx = ones(Int64, na, ns)
    pol = fill(prim.a_min, na, ns)

    # μ⁰: uniform over assets within each s, scaled by π*(s)
    π_star = stationary_earnings(prim)
    μ = repeat(π_star', na, 1) ./ na

    res = Results(v, pol_idx, pol, μ, q0)
    return prim, res
end


# ═══════════════════════════════════════════════════════════════
# 4. VALUE FUNCTION ITERATION
# ═══════════════════════════════════════════════════════════════


# One application of the Bellman operator T; returns v_next and fills pol_idx/pol
function Bellman(prim::Primitives, res::Results)
    @unpack β, α, S, ns, Π, na, a_grid = prim
    q = res.q
    v_next = zeros(na, ns)

    for is in 1:ns
        choice_lower = 1                       # Never search below the last optimum, because policy is increasing in a
        for ia in 1:na
            cash = S[is] + a_grid[ia]          # earnings + maturing bonds
            best_val = -Inf
            best_j = choice_lower

            for ja in choice_lower:na
                c = cash - q * a_grid[ja]      # bonds cost q today
                c <= 0.0 && break              # c decreasing in ja, so nothing above is feasible

                EV = 0.0
                @inbounds for isp in 1:ns
                    EV += Π[is, isp] * res.v[ja, isp]
                end

                val = u(c, α) + β * EV
                if val > best_val
                    best_val = val
                    best_j = ja
                    choice_lower = ja
                end
            end

            v_next[ia, is] = best_val
            res.pol_idx[ia, is] = best_j
            res.pol[ia, is] = a_grid[best_j]
        end
    end
    return v_next
end

# Iterate T to its fixed point, updating res.v in place
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, maxiter::Int64 = 10000)
    for iter in 1:maxiter
        v_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.v))
        res.v = v_next
        err < tol && return iter
    end
    @warn "VFI did not converge in $maxiter iterations"
    return maxiter
end

# ═══════════════════════════════════════════════════════════════
# 4. STATIONARY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

# One application of T*
function Tstar(prim::Primitives, res::Results)
    @unpack ns, na, Π = prim
    μ_next = zeros(na, ns)

    for is in 1:ns, ia in 1:na
        mass = res.μ[ia, is]
        mass == 0.0 && continue
        ia_p = res.pol_idx[ia, is]            # everyone at (a,s) moves to this asset index
        for isp in 1:ns
            μ_next[ia_p, isp] += Π[is, isp] * mass
        end
    end
    return μ_next
end

function μ_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-9, maxiter::Int64 = 100_000)
    for iter in 1:maxiter
        μ_next = Tstar(prim, res)
        err = maximum(abs.(μ_next .- res.μ))
        res.μ = μ_next
        err < tol && return iter
    end
    @warn "Distribution did not converge in $maxiter iterations"
    return maxiter
end


# ═══════════════════════════════════════════════════════════════
# 5. GENERAL EQUILIBRIUM
# ═══════════════════════════════════════════════════════════════

# Aggregate net bond demand at the current q
excess_demand(prim::Primitives, res::Results) = sum(res.pol .* res.μ)

# Bisection on q ∈ (β, 1]: solve HH problem, get μ, check market clearing
function solve_q(prim::Primitives, res::Results;
                 tol::Float64 = 1e-3, maxiter::Int64 = 50)
    q_lo, q_hi = prim.β, 1.0

    for iter in 1:maxiter
        V_iterate(prim, res)                  # step 1: decision rule at this q
        μ_iterate(prim, res)                  # step 2: invariant distribution
        ed = excess_demand(prim, res)         # step 3: market clearing

        @printf("iter %2d | q = %.6f | ED = %+.6f\n", iter, res.q, ed)
        abs(ed) < tol && return res.q

        if q_hi - q_lo < 1e-10
            @printf("bracket collapsed at q = %.8f, ED = %+.2e\n", res.q, ed)
            return res.q
        end


        # ED > 0: too much saving, bonds too cheap, raise q (lower r)
        ed > 0 ? (q_lo = res.q) : (q_hi = res.q)
        res.q = (q_lo + q_hi) / 2
    end

    @warn "Market clearing not reached in $maxiter iterations"
    return res.q
end


# ═══════════════════════════════════════════════════════════════
# 6. WEALTH DISTRIBUTION: LORENZ AND GINI
# ═══════════════════════════════════════════════════════════════

# PS definition of wealth: current earnings + net assets (avoids dividing by
# zero, since market clearing makes aggregate assets exactly 0)
function wealth_grid(prim::Primitives)
    @unpack S, ns, na, a_grid = prim
    return [a_grid[ia] + S[is] for ia in 1:na, is in 1:ns]
end

# Lorenz curve and Gini over the μ-weighted wealth distribution.
# Returns (population share, wealth share, gini).
function lorenz_gini(prim::Primitives, res::Results)
    w = vec(wealth_grid(prim))
    m = vec(res.μ)

    ord = sortperm(w)                          # Lorenz needs states ordered from poorest to richest
    w, m = w[ord], m[ord]

    pop = cumsum(m)                            # cumulative population share
    wealth = cumsum(w .* m)
    wealth ./= wealth[end]                     # cumulative wealth share

    # Gini = 1 - 2 * area under Lorenz, by the trapezoid rule
    area = 0.0
    for i in 2:length(pop)
        area += (pop[i] - pop[i-1]) * (wealth[i] + wealth[i-1]) / 2
    end
    gini = 1.0 - 2.0 * area

    return pop, wealth, gini
end

# ═══════════════════════════════════════════════════════════════
# 7. WELFARE: CONSUMPTION EQUIVALENTS
# ═══════════════════════════════════════════════════════════════

# Part I: with full insurance and no initial assets everyone consumes mean
# earnings forever, so W^FB is just that constant stream valued at β.
function welfare_complete(prim::Primitives)
    @unpack β, α, S = prim
    c_fb = dot(stationary_earnings(prim), S)
    W_fb = (c_fb^(1 - α) - 1) / ((1 - α) * (1 - β))
    return W_fb, c_fb
end

# λ(a,s): fraction of consumption an incomplete-markets household would pay
# (if > 0) to move to the complete-markets allocation
function lambda_ce(prim::Primitives, res::Results, W_fb::Float64)
    @unpack α, β = prim
    k = 1 / ((1 - α) * (1 - β))                # the constant that appears in both numerator and denominator
    return ((W_fb + k) ./ (res.v .+ k)).^(1 / (1 - α)) .- 1
end

# W^INC, the economywide gain WG, and the share of households that would favor
# switching to complete markets
function welfare_stats(prim::Primitives, res::Results)
    W_fb, c_fb = welfare_complete(prim)
    λ = lambda_ce(prim, res, W_fb)

    W_inc = sum(res.μ .* res.v)
    WG = sum(res.μ .* λ)
    share_favor = sum(res.μ[λ .>= 0.0])

    return (λ = λ, W_fb = W_fb, c_fb = c_fb, W_inc = W_inc, WG = WG, share_favor = share_favor)
end
