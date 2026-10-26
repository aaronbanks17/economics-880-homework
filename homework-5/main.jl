# import overlapping generations model methods
include(joinpath(@__DIR__, "CKmodelSS.jl"))

using Plots, Printf

# initialize plotting backend
gr()

# create output directory in current working directory
output_dir = joinpath(pwd(), "output")
mkpath(output_dir)


# ---------------------------------------
# Initialize primitives and prices
# ---------------------------------------

# initialize benchmark model
model = Model()

# initialize partial-equilibrium prices from the problem set
p_fixed = (r=0.05, w=1.05, b=0.20)


# ---------------------------------------
# Function to report steady-state results
# ---------------------------------------

function report_equilibrium(
    label::String,
    m::Model,
    results
)
    """report equilibrium aggregates, welfare, and market-clearing error"""

    # print experiment name
    println("\n", "="^60)
    println(label)
    println("="^60)

    # compute welfare and wealth inequality
    W = welfare(results.F, results.V)
    cv = wealth_cv(m, results.F)

    # compute newborn expected lifetime welfare
    W_newborn = sum(m.z_newborn .* results.V[1, :, 1])

    # report aggregate quantities and factor prices
    @printf("\nCapital K:            %.6f\n", results.K)
    @printf("Labor L:              %.6f\n", results.L)
    @printf("Wage w:               %.6f\n", results.w)
    @printf("Interest rate r:      %.6f\n", results.r)
    @printf("Pension benefit b:    %.6f\n", results.b)
    @printf("Output Y:             %.6f\n", results.K^m.α * results.L^(1 - m.α))

    # report welfare and inequality
    @printf("\nTotal welfare W:      %.6f\n", W)
    @printf("Newborn welfare:      %.6f\n", W_newborn)
    @printf("CV of wealth:         %.6f\n", cv)

    # report convergence diagnostic
    @printf("\nMarket-clearing gap:  %.6e\n", results.gap)
    if results.gap > 1e-4
        @warn "Equilibrium returned with a market-clearing gap above tolerance" label gap=results.gap
    end

    return nothing
end


# ---------------------------------------
# Function to compare policy experiments
# ---------------------------------------

function report_reform(
    label::String,
    m::Model,
    with_ss,
    without_ss
)
    """compare stationary equilibria with and without social security"""

    # print experiment name
    println("\n", "="^60)
    println("Exercise 3: $label - social security reform")
    println("="^60)

    # calculate aggregate percentage changes
    @printf("\nChange in capital K:       %+.4f%%\n", 100 * (without_ss.K / with_ss.K - 1))
    @printf("Change in labor L:         %+.4f%%\n", 100 * (without_ss.L / with_ss.L - 1))
    @printf("Change in output Y:        %+.4f%%\n", 100 * ((without_ss.K / with_ss.K)^m.α * (without_ss.L / with_ss.L)^(1 - m.α) - 1))

    # compare aggregate welfare and wealth inequality
    W_ss = welfare(with_ss.F, with_ss.V)
    W_no = welfare(without_ss.F, without_ss.V)
    cv_ss = wealth_cv(m, with_ss.F)
    cv_no = wealth_cv(m, without_ss.F)

    @printf("\nChange in total welfare W: %+.6f\n", W_no - W_ss)
    @printf("Change in CV of wealth:    %+.6f\n", cv_no - cv_ss)

    # compare newborn welfare by initial productivity type
    println("\nChange in newborn lifetime welfare by productivity:")
    for i in eachindex(m.z)
        @printf("  z = %.2f: %+.6f\n", m.z[i], without_ss.V[1, i, 1] - with_ss.V[1, i, 1])
    end

    # report welfare averaged over newborn productivity
    W_newborn_ss = sum(m.z_newborn .* with_ss.V[1, :, 1])
    W_newborn_no = sum(m.z_newborn .* without_ss.V[1, :, 1])
    @printf("\nChange in expected newborn welfare: %+.6f\n", W_newborn_no - W_newborn_ss)

    # distinguish stationary comparisons from the transition after a reform
    println("\nNote: Welfare changes compare steady states, not transition dynamics.")

    return nothing
end


# ---------------------------------------
# Function to save equilibrium results table
# ---------------------------------------

function save_results_table(
    results::Dict{String, Any},
    labels::Vector{String},
    filename::String
)
    """save Table 1 of the problem set as a LaTeX table"""

    # format one row using the requested experiment order
    function write_row(io, name::String, statistic::Function)
        values = [@sprintf("%.4f", statistic(results[label].model, results[label].equilibrium)) for label in labels]
        println(io, name, " & ", join(values, " & "), raw" \\")
    end

    # create LaTeX table
    open(joinpath(output_dir, filename), "w") do io
        println(io, raw"\begin{table}[htbp]")
        println(io, raw"\centering")
        println(io, raw"\caption{Steady-state effects of eliminating social security}")
        println(io, raw"\label{tab:ss_experiments}")
        println(io, raw"\begin{tabular}{lrrrrrr}")
        println(io, raw"\hline\hline")
        println(io, raw" & \multicolumn{2}{c}{Benchmark model} & \multicolumn{2}{c}{No idiosyncratic risk} & \multicolumn{2}{c}{Exogenous labor} \\")
        println(io, raw" & With SS & Without SS & With SS & Without SS & With SS & Without SS \\")
        println(io, raw"\hline")

        # report aggregate variables from Table 1
        write_row(io, raw"Capital $K$", (m, eq) -> eq.K)
        write_row(io, raw"Labor $L$", (m, eq) -> eq.L)
        write_row(io, raw"Wage $w$", (m, eq) -> eq.w)
        write_row(io, raw"Interest $r$", (m, eq) -> eq.r)
        write_row(io, raw"Pension benefit $b$", (m, eq) -> eq.b)
        write_row(io, raw"Total welfare $W$", (m, eq) -> welfare(eq.F, eq.V))
        write_row(io, raw"$\mathrm{cv}(\mathrm{wealth})$", (m, eq) -> wealth_cv(m, eq.F))

        # finish table
        println(io, raw"\hline\hline")
        println(io, raw"\end{tabular}")
        println(io, raw"\end{table}")
    end

    println("\nSaved LaTeX table: ", joinpath(output_dir, filename))

    return nothing
end


# ---------------------------------------
# Exercise 1: Household problem
# ---------------------------------------

# solve household problem at the given prices
household1 = solve_household(model, p_fixed)

# extract value function for a retiree at age 50
age_retired = 50
V50 = household1.V[:, 1, age_retired]

# extract savings policies for a worker at age 20
age_worker = 20
savings_high = model.a_grid[household1.policy[:, 1, age_worker]]
savings_low = model.a_grid[household1.policy[:, 2, age_worker]]

# check monotonicity and concavity on the asset grid
value_increasing = all(diff(V50) .>= -1e-8)
value_slopes = diff(V50) ./ diff(model.a_grid)
concavity_violations = count(x -> x > 1e-6, diff(value_slopes))
savings_increasing_high = all(diff(savings_high) .>= -1e-8)
savings_increasing_low = all(diff(savings_low) .>= -1e-8)
savings_increasing_z = all(savings_high .>= savings_low .- 1e-8)

# report properties of value and savings functions
println("\n", "="^60)
println("Exercise 1: Household problem")
println("="^60)
println("\nFixed prices: ", p_fixed)
println("\nRetiree value increasing in assets: ", value_increasing)
println("Retiree value concavity violations (discrete slopes): ", concavity_violations)
println("\nWorker savings increasing in assets (high z): ", savings_increasing_high)
println("Worker savings increasing in assets (low z):  ", savings_increasing_low)
println("Worker savings weakly higher for high z:   ", savings_increasing_z)

# plot retired agent value function
p_value = plot(
    model.a_grid,
    V50;
    xlabel="Assets a",
    ylabel="Value V(a)",
    title="Retiree value function, age 50",
    color=:black,
    linewidth=2,
    legend=false,
    size=(750, 500)
)

# save value function plot as PDF
savefig(p_value, joinpath(output_dir, "p1_retiree_value.pdf"))

# plot worker savings functions by productivity type
p_savings = plot(
    model.a_grid,
    savings_high;
    xlabel="Current assets a",
    ylabel="Next-period assets a'",
    label="High productivity (z = 3.0)",
    color=:black,
    linewidth=2,
    size=(750, 500)
)

plot!(
    p_savings,
    model.a_grid,
    savings_low;
    label="Low productivity (z = 0.5)",
    color=:gray,
    linestyle=:dash,
    linewidth=2
)

# save savings function plot as PDF
savefig(p_savings, joinpath(output_dir, "p1_worker_savings.pdf"))


# ---------------------------------------
# Exercise 2: Stationary distribution
# ---------------------------------------

# compute cohort sizes and stationary distribution
cohorts = cohort_sizes(model)
F2 = distribution(model, household1.policy)

# calculate cohort masses and mean assets by age
mass_by_age = [sum(F2[:, :, j]) for j in 1:model.N]
assets_by_age = [sum(F2[k, i, j] * model.a_grid[k] for k in eachindex(model.a_grid), i in eachindex(model.z)) / cohorts[j] for j in 1:model.N]

# calculate aggregate capital and labor at the fixed prices
K2, L2 = aggregates(model, F2, household1.labor)

# report distribution diagnostics
println("\n", "="^60)
println("Exercise 2: Stationary distribution")
println("="^60)
@printf("\nTotal mass:              %.8f\n", sum(F2))
@printf("Maximum cohort mass gap: %.8e\n", maximum(abs.(mass_by_age .- cohorts)))
@printf("Newborn high-z mass:     %.8f\n", F2[1, 1, 1])
@printf("Newborn low-z mass:      %.8f\n", F2[1, 2, 1])
@printf("\nAggregate assets K:      %.6f\n", K2)
@printf("Aggregate labor L:       %.6f\n", L2)
println("(Aggregates in Exercise 2 are evaluated at the fixed prices from Exercise 1.)")

# plot mean assets over the life cycle
p_assets = plot(
    1:model.N,
    assets_by_age;
    xlabel="Model age",
    ylabel="Mean assets",
    title="Stationary mean assets by age",
    color=:black,
    linewidth=2,
    legend=false,
    size=(750, 500)
)

# mark retirement age and save plot
vline!(p_assets, [model.J_R]; color=:gray, linestyle=:dash, label=false)
savefig(p_assets, joinpath(output_dir, "p2_assets_by_age.pdf"))


# ---------------------------------------
# Exercise 3: General equilibrium experiments
# ---------------------------------------

# initialize six policy experiments in the order of Table 1
experiments = [
    ("Benchmark with SS", Model()),
    ("Benchmark without SS", Model(θ=0.0)),
    ("No risk with SS", Model(z=[0.5, 0.5])),
    ("No risk without SS", Model(z=[0.5, 0.5], θ=0.0)),
    ("Exogenous labor with SS", Model(γ=1.0)),
    ("Exogenous labor without SS", Model(γ=1.0, θ=0.0))
]

# initialize results and aggregate guesses
results3 = Dict{String, Any}()
K_guess, L_guess = 3.0, 0.3

# solve for the stationary equilibrium of each experiment
for (label, m) in experiments

    # compute equilibrium, using the previous solution as the next initial guess
    println("\nSolving: ", label)
    eq = solve_equilibrium(m; K=K_guess, L=L_guess)

    # store model and equilibrium results
    results3[label] = (model=m, equilibrium=eq)

    # report results
    report_equilibrium(label, m, eq)

    # update aggregate guesses
    K_guess, L_guess = eq.K, eq.L
end

# unpack equilibrium results for each comparison
benchmark_ss = results3["Benchmark with SS"].equilibrium
benchmark_no = results3["Benchmark without SS"].equilibrium
norisk_ss = results3["No risk with SS"].equilibrium
norisk_no = results3["No risk without SS"].equilibrium
exolabor_ss = results3["Exogenous labor with SS"].equilibrium
exolabor_no = results3["Exogenous labor without SS"].equilibrium

# compare the benchmark interest rate with the implicit social security return
println("\n", "="^60)
println("Exercise 3: Dynamic efficiency")
println("="^60)
@printf("\nBenchmark interest rate r: %.6f\n", benchmark_ss.r)
@printf("Population growth rate n:  %.6f\n", model.n)
println("Dynamically efficient (r > n): ", benchmark_ss.r > model.n)

# compare benchmark social security reform
report_reform(
    "Benchmark model", model,
    benchmark_ss, benchmark_no
)

# compare capital accumulation with and without idiosyncratic risk
println("\n", "="^60)
println("Exercise 3: Precautionary savings")
println("="^60)
@printf("\nCapital with risk and SS:    %.6f\n", benchmark_ss.K)
@printf("Capital without risk and SS: %.6f\n", norisk_ss.K)
@printf("Difference (no risk - risk): %+.6f\n", norisk_ss.K - benchmark_ss.K)

# compare social security reform without idiosyncratic risk
report_reform(
    "No idiosyncratic risk", results3["No risk with SS"].model,
    norisk_ss, norisk_no
)

# compare social security reform with exogenous labor
report_reform(
    "Exogenous labor", results3["Exogenous labor with SS"].model,
    exolabor_ss, exolabor_no
)

# report changes in the labor response to removing social security
println("\n", "="^60)
println("Exercise 3: Labor-supply distortion")
println("="^60)
@printf("\nEndogenous-labor change in L: %+.6f\n", benchmark_no.L - benchmark_ss.L)
@printf("Exogenous-labor change in L: %+.6f\n", exolabor_no.L - exolabor_ss.L)

# save Table 1 to a LaTeX file
experiment_labels = [label for (label, m) in experiments]
save_results_table(results3, experiment_labels, "p3_results_table.tex")

# report output location
println("\nAll figures and tables saved in: ", output_dir)
