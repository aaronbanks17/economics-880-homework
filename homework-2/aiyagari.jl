#=
    ══════════════════════════════════════════════════════════════
    Aiyagari (1994) Model — Numerical Problem Set Solution
    Economics 714
    ══════════════════════════════════════════════════════════════
=#

using LinearAlgebra
using Printf
using Plots
using Random

Random.seed!(42)  # For reproducibility


]

# ═══════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

"""Given c, calculate CRRA utility"""
function u(c::Float64)
    c <= 0.0 && return -1e10 # Penalty for non-positive consumption
    return (c^(1 - σ_u) - 1) / (1 - σ_u)
end

"""Aggregate labor supply from stationary dist of Markov chain (no arguments)."""
function compute_agg_labor()
    eig = eigen(Π') # Transpose to get right eigenvectors
    idx = argmin(abs.(eig.values .- 1.0)) # Index of eigenvalue closest to 1
    π_stat = real.(eig.vectors[:, idx]) # Stationary distribution
    π_stat ./= sum(π_stat) # Normalize to sum to 1
    # Ensure non-negative
    π_stat = max.(π_stat, 0.0)
    π_stat ./= sum(π_stat)
    N = dot(π_stat, l_grid)
    return N, π_stat # Aggregate labor supply and stationary distribution
end

"""Given N and r, calculate capital demand curve from firm FOC"""
function K_demand(r::Float64, N::Float64)
    return N * (α * A / (r + δ))^(1 / (1 - α)) # Rearranged from FOC
end

"""Given N and r, calculate wage from firm FOC"""
function w_from_r(r::Float64, N::Float64)
    K = K_demand(r, N)
    return (1 - α) * A * (K / N)^α # From FOC: w = (1-α) A (K/N)^α
end

# ═══════════════════════════════════════════════════════════════
# 3. VALUE FUNCTION ITERATION
# ═══════════════════════════════════════════════════════════════

"""
Solve the household problem via discrete VFI.
Returns (V, pol_idx) where pol_idx[il, ia] is the index of optimal a'.
"""

# Each VFI iteration takes the current guess for V and for each (l,a) pair,
# computes the value of all feasible choices of a' on the grid and picks the 
# one that maximizes the Bellman equation. This converges to the true value 
# function and policy function as we iterate. The policy function is stored as 
# indices of the asset grid, which we can later convert to actual asset levels.

function vfi(r::Float64, w::Float64, a_grid::Vector{Float64}; # Take r, w, asset grid as inputs (i.e., solve value function for given prices)
             tol::Float64=1e-6, maxiter::Int=5000) # Tolerance and max iterations for convergence
    n_a = length(a_grid) # Asset grid size (we are discretizing the continuous choice set [-phi,40] into n_a points)

    # Initialize value function (guess: consume everything forever)
    V = zeros(n_l, n_a) # V is a matrix of size (n_l = 5, n_a) representing the value function for each labor and asset pair
    for il in 1:n_l, ia in 1:n_a # Create initial guess for V by assuming consume all income forever (no saving)
        c_approx = max(w * l_grid[il] + r * a_grid[ia], 1e-10) # Approximate consumption if consume all income (current labor income + return on assets)
        V[il, ia] = u(c_approx) / (1 - β) # Value of consuming c_approx forever (geometric series sum)
    end

    V_new = similar(V) # Temporary array to store updated value function each iteration
    pol   = zeros(Int, n_l, n_a) # Policy function indices (stores index of optimal a' for each (l,a))

    for iter in 1:maxiter # Main VFI loop
        for il in 1:n_l # Loop over labor states
            for ia in 1:n_a # Loop over asset states
                cash = w * l_grid[il] + (1 + r) * a_grid[ia] # Get current cash-on-hand (labor income + return on assets), given r
                best_val = -Inf # Initialize best value to negative infinity (we will maximize over choices)
                best_j   = 1 # Initialize best index to 1 (will update if we find better choice)

                for ja in 1:n_a # Loop over possible next-period asset choices (discrete grid)
                    c = cash - a_grid[ja] # Calculate consumption if choose a' = a_grid[ja]
                    c <= 0.0 && break   # feasibility check (grid is sorted ascending, so if c <= 0 for this ja, it will be <= 0 for all higher ja); saves time

                    # Expected continuation value
                    EV = 0.0 # Initialize expected value
                    @inbounds for jl in 1:n_l # Loop over possible next-period labor states to compute expected value / @inbounds tells Julia to skip bounds checking for speed
                        EV += Π[il, jl] * V[jl, ja] # Transition probability from current labor state il to next labor state jl times value of being in (jl, ja) next period
                    end

                    val = u(c) + β * EV # Total value of choosing a' = a_grid[ja]: current utility from consumption + discounted expected future value
                    if val > best_val # update best value and policy index if this choice is better than previous best
                        best_val = val
                        best_j   = ja
                    end
                end

                V_new[il, ia] = best_val # Store the best value found for this (il, ia) pair
                pol[il, ia]   = best_j # Store the index of the optimal next-period asset choice for this (il, ia) pair
            end
        end

        err = maximum(abs.(V_new .- V)) # Sup norm distance between old and new value function
        copyto!(V, V_new) # Update V with the new values for the next iteration
        if err < tol # Check for convergence
            @printf("    VFI converged: %d iters, err = %.2e\n", iter, err)
            return V, pol
        end
    end

    @warn "VFI did not converge after $maxiter iterations"
    return V, pol
end

# ═══════════════════════════════════════════════════════════════
# 4. STATIONARY DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

"""
Compute stationary distribution λ(l, a) by forward iteration
on the distribution operator T*.
"""

function stationary_dist(pol::Matrix{Int}, n_a::Int; # Take policy function and asset grid size as inputs
                         tol::Float64=1e-6, maxiter::Int=5000)
    ns = n_l * n_a # Total number of (l, a) states in the joint state space
    T  = zeros(ns, ns) # Initialize transition matrix T
    for il in 1:n_l, ia in 1:n_a # Construct the transition matrix T for the distribution iteration. T[j, i] is the probability of transitioning from state i to state j under the policy function and labor transitions.
        ja = pol[il, ia] # Get the index of the next-period asset choice from the policy function for current state (il, ia)
        i  = (ia - 1) * n_l + il # Convert (il, ia) to a single index i for the state space (column index in T)
        for jl in 1:n_l # Loop over possible next-period labor states to fill in the transition probabilities for T
            j = (ja - 1) * n_l + jl # Convert (jl, ja) to a single index j for the state space (row index in T)
            T[j, i] = Π[il, jl] # The probability of transitioning from labor state il to jl times the deterministic transition from asset index ia to ja given by the policy function
        end
    end

    λ = ones(ns) / ns # Start with a uniform distribution over states (initial guess)
    t_dist = time()
    for iter in 1:maxiter # Main loop for forward iteration to find stationary distribution
        λ_new = T * λ # Update the distribution by applying the transition matrix T to the current distribution λ
        err = maximum(abs.(λ_new .- λ)) # Maximum cell difference between old and new distribution
        λ = λ_new # Update λ for the next iteration

        if iter % 5000 == 0 # Print progress every 5000 iterations
            @printf("    Dist iter %5d | err = %.2e | %.1fs\n", iter, err, time() - t_dist)
        end
        if err < tol
            @printf("    Distribution converged: %d iters, err = %.2e (%.1fs)\n",
                    iter, err, time() - t_dist)
            return reshape(λ, n_l, n_a) # Reshape the distribution back to (n_l, n_a) matrix
        end
    end

    @warn "Distribution did not converge after $maxiter iters"
    return reshape(λ, n_l, n_a)
end


# ═══════════════════════════════════════════════════════════════
# 5. GENERAL EQUILIBRIUM (bisection on r)
# ═══════════════════════════════════════════════════════════════

"""
Solve the full GE model for a given borrowing limit ϕ.
Returns a NamedTuple with all results.
"""

# The GE solution involves guessing an interest rate r, solving the household problem via VFI to get the policy function, 
# computing the stationary distribution of agents across states, and then checking if the implied aggregate capital supply
# from households matches the capital demand from firms at that r. We use bisection to find the r that clears the capital market within a specified tolerance.

function solve_aiyagari(ϕ::Float64; # Take borrowing limit as input
                        n_a::Int=500, a_max::Float64=40.0,
                        tol_ge::Float64=1e-3, maxiter_ge::Int=60) # Convergence tolerance for market clearing and max bisection iterations
    N, π_stat = compute_agg_labor() # First, compute aggregate labor supply N and the stationary distribution of labor states π_stat
    a_grid = collect(range(-ϕ, a_max, length=n_a)) # Create asset grid from -ϕ to a_max with n_a points

    @printf("  Aggregate labor N = %.6f\n", N)
    @printf("  Asset grid: [%.1f, %.1f], %d points\n", -ϕ, a_max, n_a)

    # Bisection bounds: r ∈ (-δ, 1/β - 1)
    r_lo = -δ + 0.001 # r > -δ from the firm FOC (to ensure finite capital demand)
    r_hi = 1/β - 1 - 0.0005 # r < 1/β - 1 from the Euler equation (to ensure finite capital supply)

    local V, pol, λ, r_eq, w_eq, K_eq # Declare variables to store results from last iteration

    for iter in 1:maxiter_ge # Main loop
        r = (r_lo + r_hi) / 2 # Midpoint of current bracket for r
        Kd = K_demand(r, N) # Capital demand from firms at this r
        w  = w_from_r(r, N) # Wage from firms at this r

        @printf("  GE iter %2d: r = %.6f, w = %.4f, Kd = %.4f", iter, r, w, Kd)

        V, pol = vfi(r, w, a_grid) # Given these r and w, solve the household problem to get value function and policy function (this is a loop over the VFI iterations internally)
        λ = stationary_dist(pol, n_a; tol=1e-6, maxiter=5000) # Given the policy function for this r and w, compute the stationary distribution of agents across states (this is a loop over the distribution iterations internally)

        # Capital supply calculation
        Ks = 0.0 # Initialize aggregate capital supply
        for il in 1:n_l, ia in 1:n_a # Loop over all (l, a) pairs to compute aggregate capital supply by summing over the policy function choices weighted by the stationary distribution
            Ks += λ[il, ia] * a_grid[pol[il, ia]]
        end

        excess = Ks - Kd # Excess supply of capital
        @printf("  → Ks = %.4f, excess = %+.4f\n", Ks, excess)

        if abs(excess) < tol_ge # If market clears within tolerance, we have found the equilibrium
            r_eq = r; w_eq = w; K_eq = Ks
            @printf("\n  ★ Equilibrium: r = %.6f, K = %.4f, w = %.4f\n\n", r_eq, K_eq, w_eq)
            return (V=V, pol=pol, λ=λ, r=r_eq, w=w_eq, K=K_eq,
                    a_grid=a_grid, N=N, π_stat=π_stat)
        end

        if iter > 2 && abs(r_hi - r_lo) < 1e-6
            @printf("  ★ Equilibrium (r converged): r = %.6f, K = %.4f, w = %.4f\n", r, Kd, w)
            return (r=r, w=w, K=Kd, Ks=Ks, V=V, pol=pol, λ=λ, a_grid=a_grid)
        end

        # If Ks > Kd, too much saving and we should lower r
        excess > 0 ? (r_hi = r) : (r_lo = r)
    end

    # Return best guess if not converged
    r_eq = (r_lo + r_hi) / 2
    w_eq = w_from_r(r_eq, N)
    K_eq = K_demand(r_eq, N)
    @warn "GE did not converge. Returning last iterate."
    return (V=V, pol=pol, λ=λ, r=r_eq, w=w_eq, K=K_eq,
            a_grid=a_grid, N=N, π_stat=π_stat)
end

# ═══════════════════════════════════════════════════════════════
# 6. EXTRACT POLICY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

function extract_policies(res) # Take the result NamedTuple from solve_aiyagari and extract the actual policy functions for assets and consumption
    n_a = length(res.a_grid) # Get the size of the asset grid from the result
    pol_a = zeros(n_l, n_a) # Initialize policy function for next-period assets (a') as a matrix of zeros
    pol_c = zeros(n_l, n_a) # Initialize policy function for consumption (c) as a matrix of zeros
    for il in 1:n_l, ia in 1:n_a # Convert policy indices to actual asset levels and compute consumption from the budget constraint
        pol_a[il, ia] = res.a_grid[res.pol[il, ia]]
        pol_c[il, ia] = res.w * l_grid[il] + (1 + res.r) * res.a_grid[ia] - pol_a[il, ia]
    end
    return pol_a, pol_c
end

# ═══════════════════════════════════════════════════════════════
# 7. CAPITAL SUPPLY SCHEDULE
# ═══════════════════════════════════════════════════════════════

# Trace out the entire capital supply curve K^s(r) for Q9 by solving the household problem for many values of r and computing the implied aggregate capital supply
function capital_supply_schedule(ϕ::Float64, r_vec::Vector{Float64}; # Take borrowing limit and vector of interest rates (to evaluate) as inputs
                                 n_a::Int=300, a_max::Float64=40.0)
    N, _ = compute_agg_labor() # Compute aggregate labor supply N (same for all r since labor process is exogenous)
    a_grid = collect(range(-ϕ, a_max, length=n_a)) # Create asset grid for VFI

    Ks_vec = Float64[] # Initialize vector to store capital supply for each r
    for (i, r) in enumerate(r_vec) # Loop over each interest rate in r_vec to compute the corresponding capital supply
        w = w_from_r(r, N) # Get wage implied by this r
        @printf("  Schedule %2d/%d: r = %.4f ...", i, length(r_vec), r)

        _, pol = vfi(r, w, a_grid; tol=1e-6) # Solve household problem for this r and w to get policy function (this is a loop over the VFI iterations internally)
        λ = stationary_dist(pol, n_a; tol=1e-8) # Compute stationary distribution for this policy function (this is a loop over the distribution iterations internally)

        Ks = sum(λ[il, ia] * a_grid[pol[il, ia]] for il in 1:n_l, ia in 1:n_a) # Finally, compute aggregate capital supply by summing over the policy function choices weighted by the stationary distribution
        push!(Ks_vec, Ks) # Append this capital supply to the vector of results
        @printf(" Ks = %.4f\n", Ks)
    end

    return Ks_vec, N
end


# ═══════════════════════════════════════════════════════════════
#  MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

labels = ["l₁=$(round(l_grid[1],digits=3))" "l₂=$(round(l_grid[2],digits=3))" "l₃=$(round(l_grid[3],digits=3))" "l₄=$(round(l_grid[4],digits=3))" "l₅=$(round(l_grid[5],digits=3))"]


# ─── Solve baseline model (ϕ = 0) ──────────────────────

println("Solving baseline model (ϕ = 0) ...")
t = time()
res = solve_aiyagari(0.0)
pol_a, pol_c = extract_policies(res)
@printf("  Total time: %.1fs\n", time() - t)


# ─── Q5 — Value functions ────────────────────────────

println("Getting Value functions")
display(plot(res.a_grid, res.V',
    label=labels, xlabel="Assets (a)", ylabel="V(l, a)",
    title="Q5: Value Functions (ϕ=0)",
    lw=2, legend=:bottomright, size=(700,500)))
println("  → Increasing in a, concave in a, higher l shifts V up")

# ─── Q6 — Policy functions ──────────────────────────

println("Getting Savings policy function")
p_sav = plot(res.a_grid, pol_a',
    label=labels, xlabel="Current Assets (a)", ylabel="Next-Period Assets (a')",
    title="Q6: Savings Policy a'(l,a)",
    lw=2, legend=:topleft, size=(700,500))
plot!(res.a_grid, res.a_grid, label="45° line", ls=:dash, color=:black, lw=1)
display(p_sav)

println("Getting Consumption policy function")
display(plot(res.a_grid, pol_c',
    label=labels, xlabel="Assets (a)", ylabel="Consumption (c)",
    title="Q6: Consumption Policy c(l,a)",
    lw=2, legend=:topleft, size=(700,500)))
println("  → a' increasing in a and l; low-l agents constrained at a'=0")
println("  → c increasing in both; constrained agents consume ≈ current income")

# ─── Q7 — Equilibrium values ────────────────────────

println("Printing Equilibrium values")
r_cm = 1/β - 1
K_cm = K_demand(r_cm, res.N)
@printf("  Equilibrium:        r = %.6f,  K = %.4f,  w = %.4f\n", res.r, res.K, res.w)
@printf("  Complete-markets:   r = %.4f,   K = %.4f\n", r_cm, K_cm)
@printf("  → Precautionary savings: r < r_CM, K > K_CM\n")


# ─── Q8 — Asset distribution ────────────────────────

println("Asset distribution")
λ_a = vec(sum(res.λ, dims=1))
mean_a = dot(λ_a, res.a_grid)
var_a  = dot(λ_a, (res.a_grid .- mean_a).^2)
std_a  = sqrt(var_a)
cv_a   = std_a / mean_a
@printf("  Mean = %.4f,  Std = %.4f,  CV = %.4f\n", mean_a, std_a, cv_a)

display(bar(res.a_grid, λ_a,
    xlabel="Assets (a)", ylabel="Density",
    title="Q8: Asset Distribution (ϕ=0)",
    legend=false, bar_width=res.a_grid[2]-res.a_grid[1],
    fillalpha=0.7, color=:steelblue, size=(700,500),
    xlim=(-1, min(30.0, res.a_grid[end]))))

# Distribution by labor state
p_bystate = plot(title="Q8: Distribution by Labor State (ϕ=0)",
    xlabel="Assets (a)", ylabel="Density", size=(700,500))
colors = [:red, :orange, :green, :blue, :purple]
for il in 1:n_l
    plot!(res.a_grid, res.λ[il, :], label=labels[il],
          lw=2, color=colors[il], fillalpha=0.2, fill=0)
end
display(p_bystate)

# ─── Q9 — Supply and demand ─────────────────────────

println("Computing capital supply schedule ...")
t = time()
r_vec = collect(range(-0.03, 1/β - 1 - 0.005, length=12))
Ks_vec, N_val = capital_supply_schedule(0.0, r_vec)
Kd_vec = [K_demand(r, N_val) for r in r_vec]
@printf("  Supply schedule time: %.1fs\n", time() - t)

p_sd = plot(Ks_vec, r_vec, label="Capital Supply (Ks)", lw=2.5, color=:blue, size=(700,500))
plot!(Kd_vec, r_vec, label="Capital Demand (Kd)", lw=2.5, ls=:dash, color=:red)
scatter!([res.K], [res.r], label="Equilibrium", ms=7, color=:black, markershape=:star5)
hline!([1/β - 1], label="r = 1/β - 1", ls=:dot, color=:gray, lw=1)
xlabel!("Capital (K)"); ylabel!("Interest Rate (r)")
title!("Q9: Supply and Demand for Capital (ϕ=0)")
display(p_sd)

# ─── Q10 — Looser borrowing ϕ = 4 ───────────────────

println("Solving model with ϕ = 4 ...")
t = time()
res4 = solve_aiyagari(4.0)
pol_a4, pol_c4 = extract_policies(res4)
@printf("  Total time: %.1fs\n", time() - t)

@printf("\n  ϕ=0:  r = %.6f,  K = %.4f,  w = %.4f\n", res.r, res.K, res.w)
@printf("  ϕ=4:  r = %.6f,  K = %.4f,  w = %.4f\n", res4.r, res4.K, res4.w)

display(plot(res4.a_grid, res4.V',
    label=labels, xlabel="Assets (a)", ylabel="V(l, a)",
    title="Q10: Value Functions (ϕ=4)",
    lw=2, legend=:bottomright, size=(700,500)))

p_sav4 = plot(res4.a_grid, pol_a4',
    label=labels, xlabel="Current Assets (a)", ylabel="a'",
    title="Q10: Savings Policy (ϕ=4)",
    lw=2, legend=:topleft, size=(700,500))
plot!(res4.a_grid, res4.a_grid, label="45° line", ls=:dash, color=:black, lw=1)
display(p_sav4)

λ_a4 = vec(sum(res4.λ, dims=1))
mean_a4 = dot(λ_a4, res4.a_grid)
var_a4  = dot(λ_a4, (res4.a_grid .- mean_a4).^2)
@printf("  ϕ=4: Mean = %.4f, CV = %.4f\n", mean_a4, sqrt(var_a4)/abs(mean_a4))

display(bar(res4.a_grid, λ_a4,
    xlabel="Assets (a)", ylabel="Density",
    title="Q10: Asset Distribution (ϕ=4)",
    legend=false, bar_width=res4.a_grid[2]-res4.a_grid[1],
    fillalpha=0.7, color=:coral, size=(700,500),
    xlim=(-5, 30)))

println("\n  Intuition: looser borrowing → better self-insurance →")
println("  reduced precautionary motive → r closer to 1/β-1, K lower")