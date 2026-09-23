
include("hugget.jl")
using Plots, Printf

# initialize model
prim, res = initialize()

# solve for general equilibrium
@time solve_GE(prim, res)

# create directory for figures
out_dir = "output"
mkpath(out_dir)


# -- recover policy function in asset levels -- #

# recall that pol_func stores indices rather than asset levels
g = zeros(prim.n_a, prim.n_s)

for s_ix in 1:prim.n_s
    for a_ix in 1:prim.n_a
        ap_ix = res.pol_func[a_ix, s_ix]
        g[a_ix, s_ix] = prim.a_grid[ap_ix]
    end
end


# -- report equilibrium objects -- #

# calculate aggregate asset demand
asset_demand = sum(g .* res.sta_dist)

@printf("Equilibrium bond price: %.8f\n", res.q)
@printf("Aggregate asset demand: %.8e\n", asset_demand)

# report stationary probability of each income state
for s_ix in 1:prim.n_s
    state_mass = sum(res.sta_dist[:, s_ix])

    @printf(
        "Stationary mass at s = %.2f: %.6f\n",
        prim.s_grid[s_ix],
        state_mass
    )
end


# -- part II(4a): plot policy functions -- #

p_policy = plot(
    xlabel="Current assets, a",
    ylabel="Next-period assets, a'",
    title="Policy Functions",
    legend=:topleft
)

# plot policy function for each income state
for s_ix in 1:prim.n_s
    plot!(
        p_policy,
        prim.a_grid,
        g[:, s_ix],
        label="s = $(prim.s_grid[s_ix])",
        linewidth=2
    )
end

# add 45-degree line
plot!(
    p_policy,
    prim.a_grid,
    prim.a_grid,
    label="45-degree line",
    linestyle=:dash,
    linewidth=2
)

display(p_policy)

savefig(
    p_policy,
    joinpath(out_dir, "policy_functions.png")
)

# check for asset levels where g(a, s) < a
for s_ix in 1:prim.n_s

    below_45_ix = findfirst(
        g[:, s_ix] .< prim.a_grid
    )

    if isnothing(below_45_ix)

        @printf(
            "No point found where g(a, %.2f) < a.\n",
            prim.s_grid[s_ix]
        )

    else

        @printf(
            "For s = %.2f, g(a,s) < a beginning at about a = %.4f.\n",
            prim.s_grid[s_ix],
            prim.a_grid[below_45_ix]
        )
    end
end


# -- part II(4b): plot wealth distributions -- #

p_wealth = plot(
    xlabel="Wealth, w = s + a",
    ylabel="Conditional probability mass",
    title="Cross-Sectional Wealth Distributions",
    legend=:topright
)

# plot distribution conditional on each income state
for s_ix in 1:prim.n_s

# wealth is current earnings plus net assets
    wealth_s = prim.s_grid[s_ix] .+ prim.a_grid

# normalize to obtain distribution conditional on income state
    state_mass = sum(res.sta_dist[:, s_ix])

    conditional_dist =
        res.sta_dist[:, s_ix] ./ state_mass

    plot!(
        p_wealth,
        wealth_s,
        conditional_dist,
        label="s = $(prim.s_grid[s_ix])",
        linewidth=2
    )
end

display(p_wealth)

savefig(
    p_wealth,
    joinpath(out_dir, "wealth_distributions.png")
)


# -- part II(4c): Lorenz curve and Gini coefficient -- #

# collect wealth levels and probability masses across all states
wealth = Float64[]
mass = Float64[]

for s_ix in 1:prim.n_s
    for a_ix in 1:prim.n_a

# wealth is current earnings plus net assets
        w = prim.s_grid[s_ix] +
            prim.a_grid[a_ix]

# corresponding population mass
        μ = res.sta_dist[a_ix, s_ix]

        push!(wealth, w)
        push!(mass, μ)
    end
end

# sort population from lowest to highest wealth
sort_ix = sortperm(wealth)

wealth_sorted = wealth[sort_ix]
mass_sorted = mass[sort_ix]

# normalize probability mass
mass_sorted ./= sum(mass_sorted)

# calculate aggregate wealth
total_wealth = sum(
    wealth_sorted .* mass_sorted
)

# check that mean wealth is positive
if total_wealth <= 0
    error("Mean wealth must be positive to construct Lorenz curve.")
end

# cumulative population shares
cum_pop = vcat(
    0.0,
    cumsum(mass_sorted)
)

# cumulative wealth shares
cum_wealth = vcat(
    0.0,
    cumsum(
        wealth_sorted .* mass_sorted
    ) ./ total_wealth
)

# area under Lorenz curve using trapezoids
lorenz_area = sum(
    (
        cum_wealth[1:end-1] .+
        cum_wealth[2:end]
    ) .*
    diff(cum_pop) ./ 2
)

# Gini coefficient
gini = 1.0 - 2.0 * lorenz_area

@printf("Mean wealth: %.6f\n", total_wealth)
@printf("Gini coefficient: %.6f\n", gini)


# plot Lorenz curve
p_lorenz = plot(
    cum_pop,
    cum_wealth,
    label="Model",
    xlabel="Cumulative population share",
    ylabel="Cumulative wealth share",
    title="Lorenz Curve",
    linewidth=2,
    legend=:topleft
)

# add 45-degree equality line
plot!(
    p_lorenz,
    [0.0, 1.0],
    [0.0, 1.0],
    label="Perfect equality",
    linestyle=:dash,
    linewidth=2
)

display(p_lorenz)

savefig(
    p_lorenz,
    joinpath(out_dir, "lorenz_curve.png")
)