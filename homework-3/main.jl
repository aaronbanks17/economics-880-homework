using Printf, Plots

# ==========================================
# 1. Load model specifications
# ==========================================

module StandardHR
    using LinearAlgebra, Roots
    include("hopenhayn-rogerson-standard.jl")
end

module RandomHR
    using LinearAlgebra, Roots
    include("hopenhayn-rogerson-random.jl")
end


# ==========================================
# 2. Function to calculate model moments
# ==========================================

function compute_moments(res, prim)

    # mass of operating firms by productivity
    firms = res.sta_dist .+ res.M .* prim.ν

    # firm masses
    mass_incumbents = sum(res.sta_dist)
    mass_entrants = res.M
    mass_exits = sum(firms .* res.exit_pol)

    # aggregate labor
    labor_incumbents = sum(res.sta_dist .* res.labor_pol)
    labor_entrants = res.M * sum(prim.ν .* res.labor_pol)
    labor_total = labor_incumbents + labor_entrants

    # fraction of labor employed by entrants
    entrant_labor_share = labor_entrants / labor_total

    return (
        price = res.p,
        mass_incumbents = mass_incumbents,
        mass_entrants = mass_entrants,
        mass_exits = mass_exits,
        labor_total = labor_total,
        labor_incumbents = labor_incumbents,
        labor_entrants = labor_entrants,
        entrant_labor_share = entrant_labor_share
    )
end


# ==========================================
# 3. Solve standard HR model
# ==========================================

prim_std, res_std = StandardHR.initialize()

StandardHR.solve_prices!(res_std, prim_std)
StandardHR.solve_labor_market!(res_std, prim_std)

mom_std = compute_moments(res_std, prim_std)


# ==========================================
# 4. Solve TV1 model: alpha = 1
# ==========================================

prim_tv1, res_tv1 = RandomHR.initialize(α = 1.0)

RandomHR.solve_prices!(res_tv1, prim_tv1)
RandomHR.solve_labor_market!(res_tv1, prim_tv1)

mom_tv1 = compute_moments(res_tv1, prim_tv1)


# ==========================================
# 5. Solve TV1 model: alpha = 2
# ==========================================

prim_tv2, res_tv2 = RandomHR.initialize(α = 2.0)

RandomHR.solve_prices!(res_tv2, prim_tv2)
RandomHR.solve_labor_market!(res_tv2, prim_tv2)

mom_tv2 = compute_moments(res_tv2, prim_tv2)


# ==========================================
# 6. Print equilibrium comparison
# ==========================================

println("\n================ EQUILIBRIUM COMPARISON ================\n")

@printf(
    "%-35s %15s %15s %15s\n",
    "Variable", "Standard", "TV1 α = 1", "TV1 α = 2"
)

println("-"^85)

function print_row(label, x1, x2, x3)
    @printf("%-35s %15.6f %15.6f %15.6f\n", label, x1, x2, x3)
end

print_row(
    "Price Level",
    mom_std.price, mom_tv1.price, mom_tv2.price
)

print_row(
    "Mass of Incumbents",
    mom_std.mass_incumbents, mom_tv1.mass_incumbents, mom_tv2.mass_incumbents
)

print_row(
    "Mass of Entrants",
    mom_std.mass_entrants, mom_tv1.mass_entrants, mom_tv2.mass_entrants
)

print_row(
    "Mass of Exits",
    mom_std.mass_exits, mom_tv1.mass_exits, mom_tv2.mass_exits
)

print_row(
    "Aggregate Labor",
    mom_std.labor_total, mom_tv1.labor_total, mom_tv2.labor_total
)

print_row(
    "Labor of Incumbents",
    mom_std.labor_incumbents, mom_tv1.labor_incumbents, mom_tv2.labor_incumbents
)

print_row(
    "Labor of Entrants",
    mom_std.labor_entrants, mom_tv1.labor_entrants, mom_tv2.labor_entrants
)

print_row(
    "Fraction of Labor in Entrants",
    mom_std.entrant_labor_share,
    mom_tv1.entrant_labor_share,
    mom_tv2.entrant_labor_share
)

# ==========================================
# 7. Set up output directory
# ==========================================

output_dir = joinpath(pwd(), "output")
mkpath(output_dir)

# productivity states and axis labels
s_grid = prim_std.s_grid
s_labels = ["0.0004", "3.58", "6.82", "12.18", "18.79"]


# ==========================================
# 8. Task 3: Compare exit decision rules
# ==========================================

# exit policies at c_f = 10
exit_std = Float64.(res_std.exit_pol)
exit_tv1 = res_tv1.exit_pol
exit_tv2 = res_tv2.exit_pol

# plot standard model
p3 = plot(
    s_grid,
    exit_std;
    label = "Standard",
    marker = :circle,
    linewidth = 2,
    markersize = 6,
    xlabel = "Productivity",
    ylabel = "Probability of Exit",
    title = "Equilibrium Exit Policies",
    xticks = (s_grid, s_labels),
    ylims = (-0.05, 1.05),
    legend = :topright,
    size = (850, 500)
)

# add TV1 alpha = 1
plot!(
    p3,
    s_grid,
    exit_tv1;
    label = "TV1 (α = 1)",
    marker = :square,
    linewidth = 2,
    markersize = 6
)

# add TV1 alpha = 2
plot!(
    p3,
    s_grid,
    exit_tv2;
    label = "TV1 (α = 2)",
    marker = :diamond,
    linewidth = 2,
    markersize = 6
)

# save figure
savefig(p3, joinpath(output_dir, "task3_exit_policies.pdf"))


# ==========================================
# 9. Task 4: Increase c_f from 10 to 15
# ==========================================

# -- Standard HR -- #

# new primitives with higher fixed cost
prim_std_15 = StandardHR.Primitives(c_f = 15.0)

# initialize results
_, res_std_15 = StandardHR.initialize()

# resolve equilibrium
StandardHR.solve_prices!(res_std_15, prim_std_15)
StandardHR.solve_labor_market!(res_std_15, prim_std_15)


# -- TV1 HR: alpha = 1 -- #

# new primitives with higher fixed cost
prim_tv1_15 = RandomHR.Primitives(
    c_f = 15.0,
    α = 1.0
)

# initialize results
_, res_tv1_15 = RandomHR.initialize(α = 1.0)

# resolve equilibrium
RandomHR.solve_prices!(res_tv1_15, prim_tv1_15)
RandomHR.solve_labor_market!(res_tv1_15, prim_tv1_15)


# -- TV1 HR: alpha = 2 -- #

# new primitives with higher fixed cost
prim_tv2_15 = RandomHR.Primitives(
    c_f = 15.0,
    α = 2.0
)

# initialize results
_, res_tv2_15 = RandomHR.initialize(α = 2.0)

# resolve equilibrium
RandomHR.solve_prices!(res_tv2_15, prim_tv2_15)
RandomHR.solve_labor_market!(res_tv2_15, prim_tv2_15)


# ==========================================
# 10. Task 4: Plot comparative statics
# ==========================================

# function to compare exit policies across fixed costs
function plot_exit_comparison(s_grid, exit_10, exit_15, title_str)

    p = plot(
        s_grid,
        Float64.(exit_10);
        label = "c_f = 10",
        marker = :circle,
        linewidth = 2,
        markersize = 5,
        xlabel = "Productivity",
        ylabel = "Probability of Exit",
        title = title_str,
        xticks = (s_grid, s_labels),
        ylims = (-0.05, 1.05),
        legend = :topright
    )

    plot!(
        p,
        s_grid,
        Float64.(exit_15);
        label = "c_f = 15",
        marker = :square,
        linewidth = 2,
        markersize = 5,
        linestyle = :dash
    )

    return p
end


# standard model
p_std = plot_exit_comparison(
    s_grid,
    res_std.exit_pol,
    res_std_15.exit_pol,
    "Standard"
)

# TV1 alpha = 1
p_tv1 = plot_exit_comparison(
    s_grid,
    res_tv1.exit_pol,
    res_tv1_15.exit_pol,
    "TV1 (α = 1)"
)

# TV1 alpha = 2
p_tv2 = plot_exit_comparison(
    s_grid,
    res_tv2.exit_pol,
    res_tv2_15.exit_pol,
    "TV1 (α = 2)"
)

# combine into one figure
p4 = plot(
    p_std,
    p_tv1,
    p_tv2;
    layout = (1, 3),
    size = (1400, 450),
    link = :y
)

# save figure
savefig(p4, joinpath(output_dir, "task4_fixed_cost_comparison.pdf"))


# ==========================================
# 11. Print comparative statics
# ==========================================

println("\n================ FIXED COST COMPARISON ================\n")

# equilibrium prices
@printf(
    "%-25s %12s %12s\n",
    "Model", "c_f = 10", "c_f = 15"
)

println("-"^51)

@printf(
    "%-25s %12.6f %12.6f\n",
    "Standard", res_std.p, res_std_15.p
)

@printf(
    "%-25s %12.6f %12.6f\n",
    "TV1 alpha = 1", res_tv1.p, res_tv1_15.p
)

@printf(
    "%-25s %12.6f %12.6f\n",
    "TV1 alpha = 2", res_tv2.p, res_tv2_15.p
)


# exit probabilities by productivity
function print_exit_comparison(s_grid, exit_10, exit_15, model_name)

    println("\n", model_name)

    @printf(
        "%15s %15s %15s\n",
        "Productivity", "c_f = 10", "c_f = 15"
    )

    println("-"^47)

    for i in eachindex(s_grid)

        @printf(
            "%15.4f %15.6f %15.6f\n",
            s_grid[i],
            Float64(exit_10[i]),
            Float64(exit_15[i])
        )

    end
end


print_exit_comparison(
    s_grid,
    res_std.exit_pol,
    res_std_15.exit_pol,
    "Standard HR"
)

print_exit_comparison(
    s_grid,
    res_tv1.exit_pol,
    res_tv1_15.exit_pol,
    "TV1 alpha = 1"
)

print_exit_comparison(
    s_grid,
    res_tv2.exit_pol,
    res_tv2_15.exit_pol,
    "TV1 alpha = 2"
)


println("\nFigures saved to: ", output_dir)