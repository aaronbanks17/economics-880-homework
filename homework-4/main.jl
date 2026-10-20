
# import SMM methods
include(joinpath(@__DIR__, "smm.jl"))

using Plots, Printf

# initialize plotting backend
gr()

# create output directory in current working directory
output_dir = joinpath(pwd(), "output")
mkpath(output_dir)


# ---------------------------------------
# Initialize primitives and data
# ---------------------------------------

# initialize primitives
prim = Primitives()

# initialize data and simulated errors
rng = MersenneTwister(42)
H = 10
data = initialize(H; rng=rng)

# initialize parameter guess
b_init = [0.5, 1.0]

# define moment specifications
m4 = (:mean, :variance)
m5 = (:variance, :autocorrelation)
m6 = (:mean, :variance, :autocorrelation)

# initialize parameter grids for objective surfaces
ρ_grid = collect(range(0.35, 0.65, length=75))
σ_grid = collect(range(0.8, 1.2, length=75))


# ---------------------------------------
# Function to report SMM results
# ---------------------------------------

function report_smm(
    problem::Int64,
    prim::Primitives,
    data::Data,
    selected::Tuple{Vararg{Symbol}},
    results
)
    """report SMM estimates and inference statistics"""

    # print problem number and moment specification
    println("\n", "="^60)
    println("Problem $problem")
    println("Moments: ", selected)
    println("="^60)

    # report empirical moments
    println("\nEmpirical moments:")
    println(results.m_data)

    # report parameter estimates
    println("\nFirst-stage estimates:")
    println(results.b1)

    println("\nSecond-stage estimates:")
    println(results.b2)

    # report objective function values
    @printf("\nFirst-stage objective:  %.8f\n", results.J1)
    @printf("Second-stage objective: %.8f\n", results.J2)

    # report optimal weighting matrix
    println("\nNewey-West covariance matrix:")
    println(results.S)

    println("\nOptimal weighting matrix:")
    println(results.W2)

    # report optimizer convergence
    println("\nFirst-stage converged: ", results.converged1)
    println("Second-stage converged: ", results.converged2)

    # compute J-test independently of other inference
    J = j_test(
        results.J2,
        prim.T,
        size(data.errors, 2),
        length(selected),
        length(results.b2)
    )

    println("\nJ-test statistic: ", J.statistic)
    println("Degrees of freedom: ", J.df)
    println("J-test p-value: ", J.pvalue)

    # compute inference statistics
    inference = try
        smm_inference(prim, data, selected, results)
    catch err
        @warn "Inference failed for Problem $problem" exception=err
        nothing
    end

    if inference !== nothing

        # report numerical jacobian
        println("\nJacobian:")
        println(inference.G)

        # report parameter variance-covariance matrix
        println("\nParameter covariance matrix:")
        println(inference.V)

        # report standard errors
        println("\nStandard errors:")
        println(inference.se)

        # report AGS sensitivity matrix
        println("\nAGS sensitivity matrix:")
        println(inference.Λ)

        # report singular values
        println("\nJacobian singular values:")
        println(inference.singular_values)

        # report identification diagnostic
        A = inference.G' * results.W2 * inference.G
        κ = cond(A)

        @printf("\nCondition number of G'WG: %.6e\n", κ)

        if !isfinite(κ) || κ > 1e10
            println("Warning: potential identification or conditioning problem.")
        end
    end

    return inference
end


# ---------------------------------------
# Function to save SMM objective surface
# ---------------------------------------

function save_smm_surface(
    prim::Primitives,
    data::Data,
    selected::Tuple{Vararg{Symbol}},
    ρ_grid::AbstractVector{Float64},
    σ_grid::AbstractVector{Float64},
    filename::String
)
    """compute and save three-dimensional SMM objective surface"""

    # compute objective over parameter grid
    J = smm_surface(
        prim, data, selected, ρ_grid, σ_grid
    )

    # construct three-dimensional surface
    p = surface(
        ρ_grid,
        σ_grid,
        J;
        xlabel="ρ",
        ylabel="σ",
        zlabel="J(b)",
        camera=(50, 30),
        legend=false,
        size=(800, 600)
    )

    # save plot as PDF
    savefig(p, joinpath(output_dir, filename))

    return nothing
end


# ---------------------------------------
# Problem 4: Mean and variance
# ---------------------------------------

# estimate parameters
results4 = estimate_smm(
    prim, data, m4, b_init
)

# compute inference and report results
inference4 = report_smm(
    4, prim, data, m4, results4
)

# generate objective surface
save_smm_surface(
    prim, data, m4, ρ_grid, σ_grid,
    "p4_surface.pdf"
)


# ---------------------------------------
# Problem 5: Variance and autocorrelation
# ---------------------------------------

# estimate parameters
results5 = estimate_smm(
    prim, data, m5, b_init
)

# compute inference and report results
inference5 = report_smm(
    5, prim, data, m5, results5
)

# generate objective surface
save_smm_surface(
    prim, data, m5, ρ_grid, σ_grid,
    "p5_surface.pdf"
)


# ---------------------------------------
# Problem 6: All three moments
# ---------------------------------------

# estimate parameters
results6 = estimate_smm(
    prim, data, m6, b_init
)

# compute inference and report results
inference6 = report_smm(
    6, prim, data, m6, results6
)

# generate objective surface
save_smm_surface(
    prim, data, m6, ρ_grid, σ_grid,
    "p6_surface.pdf"
)


# ---------------------------------------
# Problem 6: Bootstrap
# ---------------------------------------

# initialize bootstrap random number generator
rng_boot = MersenneTwister(123)

# set number of replications
B = 500

# run bootstrap
boot6 = bootstrap_smm(
    prim, H, m6, b_init;
    B=B,
    rng=rng_boot
)

# report bootstrap summary statistics
println("\n", "="^60)
println("Problem 6: Bootstrap")
println("="^60)

println("\nTrue parameters:")
println(prim.b_0)

println("\nMean first-stage estimates:")
println(boot6.mean_b1)

println("\nMean second-stage estimates:")
println(boot6.mean_b2)

println("\nStandard deviation of first-stage estimates:")
println(boot6.sd_b1)

println("\nStandard deviation of second-stage estimates:")
println(boot6.sd_b2)

println("\nBias of first-stage estimates:")
println(boot6.mean_b1 - prim.b_0)

println("\nBias of second-stage estimates:")
println(boot6.mean_b2 - prim.b_0)

println("\nConverged replications: ",
    count(boot6.converged), " / ", B
)

if !all(boot6.converged)
    @warn "Some bootstrap replications did not converge. Summary statistics include these estimates."
end


# ---------------------------------------
# Problem 6: Bootstrap comparison plots
# ---------------------------------------

function bootstrap_comparison(
    b1::AbstractVector{Float64},
    b2::AbstractVector{Float64},
    true_value::Float64,
    parameter::String,
    filename::String
)
    """compare first- and second-stage bootstrap distributions"""

    # construct common histogram bins
    lower = min(minimum(b1), minimum(b2))
    upper = max(maximum(b1), maximum(b2))
    edges = collect(range(lower, upper, length=31))

    # plot first-stage distribution
    p = histogram(
        b1;
        bins=edges,
        normalize=:pdf,
        color=:gray,
        fillalpha=0.5,
        label="First stage",
        xlabel=parameter,
        ylabel="Density",
        size=(700, 500)
    )

    # overlay second-stage distribution
    stephist!(
        p, b2;
        bins=edges,
        normalize=:pdf,
        color=:black,
        linewidth=2,
        label="Second stage"
    )

    # add true parameter value
    vline!(
        p, [true_value];
        color=:black,
        linestyle=:dash,
        linewidth=1.5,
        label="True value"
    )

    # save plot
    savefig(p, joinpath(output_dir, filename))

    return nothing
end


# compare distributions of rho
bootstrap_comparison(
    vec(boot6.b1[1, :]),
    vec(boot6.b2[1, :]),
    prim.b_0[1],
    "ρ",
    "p6_bootstrap_rho_compare.pdf"
)

# compare distributions of sigma
bootstrap_comparison(
    vec(boot6.b1[2, :]),
    vec(boot6.b2[2, :]),
    prim.b_0[2],
    "σ",
    "p6_bootstrap_sigma_compare.pdf"
)

# plot joint distribution of second-stage estimates
p = scatter(
    vec(boot6.b2[1, :]),
    vec(boot6.b2[2, :]);
    xlabel="ρ",
    ylabel="σ",
    color=:gray,
    alpha=0.5,
    markersize=3,
    markerstrokewidth=0,
    legend=false,
    size=(700, 500)
)

# mark true parameters
scatter!(
    p,
    [prim.b_0[1]],
    [prim.b_0[2]];
    color=:black,
    markershape=:star5,
    markersize=9
)

savefig(
    p,
    joinpath(output_dir, "p6_bootstrap_joint.pdf")
)

# ---------------------------------------
# Problem 7: Indirect Inference
# ---------------------------------------

# initialize results
results7 = Dict{Int64, Any}()

# estimate indirect inference for N = 1, 2, 3
for N in 1:3

    # estimate structural parameters
    results = estimate_indirect(
        prim, data, N, b_init
    )

    # store results
    results7[N] = results

    # report results
    println("\n", "="^60)
    println("Problem 7: MA($N) Indirect Inference")
    println("="^60)

    println("\nEmpirical auxiliary moments:")
    println(results.m_data)

    println("\nSimulated auxiliary moments:")
    println(results.m_sim)

    println("\nStructural parameter estimates:")
    println(results.b)

    println("\nTrue parameters:")
    println(prim.b_0)

    @printf(
        "\nObjective function: %.8f\n",
        results.objective
    )

    println("\nConverged: ", results.converged)

end

