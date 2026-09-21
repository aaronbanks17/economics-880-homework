##### Huggett (1993) — solve the model and produce all PS2 output #####

using Plots
using Printf

include("huggett_functions.jl")

# Inputs:
# 'huggett_functions.jl' <- model primitives, solver, welfare functions

# Outputs:
# 'output/policy_functions.png'    <- II.4a, g(a,s) with the 45 degree line
# 'output/wealth_distribution.png' <- II.4b, cross-sectional wealth by employment state
# 'output/lorenz.png'              <- II.4c, Lorenz curve
# 'output/lambda.png'              <- III.a, consumption equivalents

OUT_DIR = "output"

# ═══════════════════════════════════════════════════════════════
# SOLVE
# ═══════════════════════════════════════════════════════════════

prim, res = Initialize()
solve_q(prim, res)

a_grid = prim.a_grid
wealth = wealth_grid(prim)

# ═══════════════════════════════════════════════════════════════
# II.4a — POLICY FUNCTION
# ═══════════════════════════════════════════════════════════════

p_pol = plot(a_grid, res.pol[:, 1], label = "employed", lw = 2,
    xlabel = "a", ylabel = "a' = g(a,s)", title = "Policy function",
    legend = :topleft, size = (700, 500))
plot!(a_grid, res.pol[:, 2], label = "unemployed", lw = 2)
plot!(a_grid, a_grid, label = "45°", ls = :dash, color = :black, lw = 1)
savefig(p_pol, joinpath(OUT_DIR, "policy_functions.png"))

# â where g(â,s) < â: confirms the upper bound of A is not binding
a_hat = [a_grid[findfirst(i -> res.pol[i, is] < a_grid[i], 1:prim.na)] for is in 1:prim.ns]
@printf("â (employed) = %.4f | â (unemployed) = %.4f | a_max = %.1f\n",
    a_hat[1], a_hat[2], prim.a_max)
@printf("g(a_max, e) = %.4f < %.1f ✓   g(a_max, u) = %.4f < %.1f ✓\n",
    res.pol[end, 1], prim.a_max, res.pol[end, 2], prim.a_max)

# ═══════════════════════════════════════════════════════════════
# II.4b — BOND PRICE AND WEALTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════

@printf("\nq*  = %.6f   (r = %.4f%% per period)\n", res.q, 100 * (1 / res.q - 1))
@printf("β   = %.6f   (complete markets: r = %.4f%%)\n", prim.β, 100 * (1 / prim.β - 1))
@printf("ED  = %+.2e\n", excess_demand(prim, res))

p_dist = plot(a_grid, res.μ[:, 1], label = "employed", lw = 2,
    xlabel = "a", ylabel = "mass", title = "Cross-sectional distribution of wealth",
    legend = :topright, size = (700, 500))
plot!(a_grid, res.μ[:, 2], label = "unemployed", lw = 2)
savefig(p_dist, joinpath(OUT_DIR, "wealth_distribution.png"))

# ═══════════════════════════════════════════════════════════════
# II.4c — LORENZ CURVE AND GINI
# ═══════════════════════════════════════════════════════════════

pop_share, wealth_share, gini = lorenz_gini(prim, res)

p_lorenz = plot(pop_share, wealth_share, label = "Lorenz", lw = 2,
    xlabel = "cumulative share of population", ylabel = "cumulative share of wealth",
    title = @sprintf("Lorenz curve (Gini = %.4f)", gini),
    legend = :topleft, size = (700, 500))
plot!(pop_share, pop_share, label = "perfect equality", ls = :dash, color = :black, lw = 1)
savefig(p_lorenz, joinpath(OUT_DIR, "lorenz.png"))

@printf("\nGini (wealth = s + a) = %.4f\n", gini)
@printf("mean wealth = %.4f\n", sum(wealth .* res.μ))

# ═══════════════════════════════════════════════════════════════
# III — WELFARE: INCOMPLETE VS COMPLETE MARKETS
# ═══════════════════════════════════════════════════════════════

ws = welfare_stats(prim, res)

p_lambda = plot(a_grid, ws.λ[:, 1], label = "employed", lw = 2,
    xlabel = "a", ylabel = "λ(a,s)", title = "Consumption equivalents",
    legend = :topright, size = (700, 500))
plot!(a_grid, ws.λ[:, 2], label = "unemployed", lw = 2)
hline!([0.0], label = "", ls = :dash, color = :black, lw = 1)
savefig(p_lambda, joinpath(OUT_DIR, "lambda.png"))

@printf("\nc_FB  = %.4f      W_FB  = %.4f\n", ws.c_fb, ws.W_fb)
@printf("W_INC = %.4f      WG    = %.5f\n", ws.W_inc, ws.WG)
@printf("share favoring complete markets = %.4f\n", ws.share_favor)
@printf("λ range: [%.4f, %.4f]\n", minimum(ws.λ), maximum(ws.λ))
