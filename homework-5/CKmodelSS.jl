# Overlapping generations model of social security, after Conesa and Krueger (1999).
# The age-efficiency profile is read from ef.txt

"""Parameters and grids. Agents live for ages j = 1, …, N, work before age J_R, and are retired from J_R on."""
@kwdef struct Model
    N::Int = 66                                          # Length of life
    J_R::Int = 46                                        # Retirement age
    n::Float64 = 0.011                                   # Population growth rate
    β::Float64 = 0.97                                    # Discount factor
    σ::Float64 = 2.0                                     # Coefficient of relative risk aversion
    γ::Float64 = 0.42                                    # Weight on consumption (γ = 1: exogenous labor)
    θ::Float64 = 0.11                                    # Social security tax rate
    α::Float64 = 0.36                                    # Capital share
    δ::Float64 = 0.06                                    # Depreciation rate
    z::Vector{Float64} = [3.0, 0.5]                      # Idiosyncratic productivity (high, low)
    Π::Matrix{Float64} = [0.9261 0.0739; 0.0189 0.9811]  # Π[i, i′] = Pr(z′ = z_i′ | z = z_i)
    z_newborn::Vector{Float64} = [0.2037, 0.7963]        # Distribution of z at birth (ergodic for Π)
    η::Vector{Float64} = parse.(Float64, split(read(joinpath(@__DIR__, "ef.txt"), String)))  # Age efficiency, j < J_R
    a_grid::Vector{Float64} = 100 .* range(0, 1; length=2001) .^ 2  # Assets, denser near a = 0 (no borrowing)
end

utility(m::Model, c, l) = (c^m.γ * (1 - l)^(1 - m.γ))^(1 - m.σ) / (1 - m.σ)  # Equals u^R(c) when l = 0

efficiency(m::Model, i, j) = j < m.J_R ? m.z[i] * m.η[j] : 0.0  # e(z_i, j); retirees have none

"""Cohort sizes μ_j, with μ_{j+1} = μ_j / (1 + n), normalized to sum to one."""
function cohort_sizes(m::Model)
    μ = (1 + m.n) .^ -(0:m.N-1)
    return μ ./ sum(μ)
end

"""Competitive factor prices given K and L, and the benefit b that balances the pay-as-you-go budget."""
function prices(m::Model, K, L)
    r = m.α * (L / K)^(1 - m.α) - m.δ
    w = (1 - m.α) * (K / L)^m.α
    b = m.θ * w * L / sum(cohort_sizes(m)[m.J_R:end])
    return (; r, w, b)
end

"""
Solve the household problem by backward induction from V_{N+1} = 0. Arrays are indexed [a, z, j], and
`policy` holds grid indices of a′. One loop covers workers and retirees: an agent earns y = (1 − θ)we(z, j) per unit
of labor, and x = (1 + r)a + b − a′ is the rest of consumption (b is paid only to retirees). The labor FOC gives
l = γ − (1 − γ)x/y, so c = yl + x. Retirees have y = 0, so l = 0 and u(c, 0) = u^R(c).
Since a′(a, z) is increasing in a, each search starts at the choice made at the previous a; since the objective is
concave in a′, it stops as soon as the objective falls.
"""
function solve_household(m::Model, (; r, w, b))
    na, nz = length(m.a_grid), length(m.z)
    V = zeros(na, nz, m.N + 1)
    policy, labor = ones(Int, na, nz, m.N), zeros(na, nz, m.N)
    for j in m.N:-1:1
        EV = V[:, :, j+1] * m.Π'                         # EV[a′, i] = Σ_i′ Π[i, i′] V_{j+1}(a′, z_i′)
        pension = j < m.J_R ? 0.0 : b
        for i in 1:nz
            y = (1 - m.θ) * w * efficiency(m, i, j)
            start = 1
            for k in 1:na
                best = -Inf
                for k′ in start:na
                    x = (1 + r) * m.a_grid[k] + pension - m.a_grid[k′]
                    l = y > 0 ? clamp(m.γ - (1 - m.γ) * x / y, 0, 1) : 0.0
                    c = y * l + x
                    c > 0 || break                       # Every larger a′ is unaffordable too
                    value = utility(m, c, l) + m.β * EV[k′, i]
                    value < best && break
                    best, policy[k, i, j], labor[k, i, j] = value, k′, l
                end
                V[k, i, j] = best
                start = policy[k, i, j]
            end
        end
    end
    return (; V=V[:, :, 1:m.N], policy, labor)
end

"""
Steady-state distribution F[a, z, j]. Newborns start with a = 0 and z drawn from `z_newborn`, then follow
the policy and the z process; each cohort is 1/(1 + n) times the size of the one before. Retirees' z plays no role.
"""
function distribution(m::Model, policy)
    F = zeros(size(policy))
    F[1, :, 1] = cohort_sizes(m)[1] .* m.z_newborn
    for j in 1:m.N-1, i in eachindex(m.z), k in eachindex(m.a_grid)
        F[policy[k, i, j], :, j+1] .+= F[k, i, j] .* m.Π[i, :] ./ (1 + m.n)
    end
    return F
end

"""Aggregate capital (assets) and effective labor supplied by households."""
function aggregates(m::Model, F, labor)
    K = sum(F .* m.a_grid)
    L = sum(F[k, i, j] * efficiency(m, i, j) * labor[k, i, j] for k in axes(F, 1), i in axes(F, 2), j in axes(F, 3))
    return K, L
end

"""
Guess and verify: solve the household problem and distribution at the prices implied by the guess (K, L), then move
the guess a fraction `step` of the way toward the implied aggregates. Since choices lie on a grid, the implied
aggregates jump as prices change and there may be no exact fixed point, so the step is halved whenever the relative
gap widens. Stops once the gap or the step is below `tol`, which takes 13 to 113 iterations for the six economies in
run_solution.jl.
"""
function solve_equilibrium(m::Model; K=3.0, L=0.3, step=0.5, tol=1e-4, max_iter=1000)
    gap_old = Inf
    for _ in 1:max_iter
        p = prices(m, K, L)
        household = solve_household(m, p)
        F = distribution(m, household.policy)
        K_new, L_new = aggregates(m, F, household.labor)
        gap = max(abs(K_new - K) / K, abs(L_new - L) / L)
        gap > gap_old && (step /= 2)
        (gap < tol || step < tol) && return (; K, L, p..., household..., F, gap)
        K, L, gap_old = (1 - step) * K + step * K_new, (1 - step) * L + step * L_new, gap
    end
    error("The equilibrium did not converge in $max_iter iterations (relative gap $gap_old, step $step).")
end

"""Aggregate welfare Σ F V (skipping empty states, where V may be −Inf)."""
welfare(F, V) = sum(F[s] * V[s] for s in eachindex(F) if F[s] > 0)

"""Coefficient of variation of wealth (assets) across all agents."""
function wealth_cv(m::Model, F)
    a = m.a_grid .+ zero(F)                              # Asset level of every state
    mean = sum(F .* a)
    return sqrt(sum(F .* (a .- mean) .^ 2)) / mean
end
