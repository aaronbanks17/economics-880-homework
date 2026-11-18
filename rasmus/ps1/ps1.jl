##### PS1 (Lentz) — Rust (1987) bus engine replacement: hazard + mileage MLE #####

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Distributions
using Optim
using Printf
using Plots
using Random
using JuMP
using Ipopt

# Inputs:
# 'busdata.csv' <- monthly bus obs

# Outputs:
# 'output/bins.png'          <- obs and replacements per bin
# 'output/hazard_binned.png' <- Q1 binned hazard
# 'output/hazard_bern.png'   <- Q2 Bernstein hazard over binned


const PS_DIR    = joinpath(homedir(), "Life", "courses", "ECON880 Computational", "homeworks", "rasmus", "ps1")
const DATA_FILE = "busdata.csv"
const N_BINS    = 40

cd(PS_DIR)
mkpath("output")


######################
#### Load data ####
######################

# Recompute delta from primo/ultimo
const COLS = [:id, :type, :year, :month, :replace,
              :odo_primo, :odo_ultimo, :odo_ultimo_total, :odo_delta]

db_bus = CSV.read(DATA_FILE, DataFrame; header = COLS)

# Calendar month index (1 = July 1975) and sort within bus
db_bus.t = (db_bus.year .- 75) .* 12 .+ db_bus.month .- 6
sort!(db_bus, [:id, :t])

describe(db_bus)


#######################################
#### Build estimation variables ####
#######################################

# Mileage increment: Δx_t = x_t - (1 - d_t) x_{t-1}
db_bus.dx = ifelse.(db_bus.replace .== 1, db_bus.odo_ultimo, db_bus.odo_ultimo .- db_bus.odo_primo)

# State at decision time and choice
db_bus.x_state = db_bus.odo_primo
db_bus.d       = db_bus.replace

# drop Δx = 0 (bus idle that month) as it has no log-normal density
db_bus.dx_pos = db_bus.dx .> 0

@printf("Rows: %d | buses: %d | replacements: %d | Δx = 0: %d | Δx < 0: %d\n",
        nrow(db_bus), length(unique(db_bus.id)), sum(db_bus.d),
        sum(db_bus.dx .== 0), sum(db_bus.dx .< 0))


##############################
#### Grid and bins ####
##############################

# Uniform grid x̄_0 = 0, ..., x̄_N = max(x) + 1, over every x value used
const X_MAX = max(maximum(db_bus.odo_primo), maximum(db_bus.odo_ultimo)) + 1
xgrid = collect(range(0, X_MAX, length = N_BINS + 1))

# Bin n such that x̄_{n-1} ≤ x < x̄_n
db_bus.bin = searchsortedlast.(Ref(xgrid), db_bus.x_state)

# Obs (K_n) and replacements (D_n) per bin
df_bins = combine(groupby(db_bus, :bin), nrow => :K, :d => sum => :D)
df_bins = leftjoin(DataFrame(bin = 1:N_BINS), df_bins, on = :bin)
sort!(df_bins, :bin)
df_bins.K = coalesce.(df_bins.K, 0)
df_bins.D = coalesce.(df_bins.D, 0)
df_bins.x_lo  = xgrid[1:N_BINS]
df_bins.x_hi  = xgrid[2:end]
df_bins.x_mid = (df_bins.x_lo .+ df_bins.x_hi) ./ 2

df_bins

# Obs (K_n) and replacements (D_n) per bin; bars span each bin's [x̄_{n-1}, x̄_n)
bin_w = xgrid[2] - xgrid[1]
p_K = bar(df_bins.x_mid ./ 1000, df_bins.K, bar_width = bin_w / 1000, legend = false,
          ylabel = "Obs per bin (K_n)", title = "Bins, N = $(N_BINS)")
p_D = bar(df_bins.x_mid ./ 1000, df_bins.D, bar_width = bin_w / 1000, legend = false,
          ylabel = "Replacements (D_n)", xlabel = "Mileage since last replacement (000s)")
p_bins = plot(p_K, p_D, layout = (2, 1), link = :x, size = (700, 550))
savefig(p_bins, "output/bins.png")


###############################
#### Q1: closed-form MLE ####
###############################

#### Mileage law of motion: ln Δx ~ N(μ, σ) ####

# FOCs: μ̂ = (1/M) Σ ln Δx ;  σ̂² = (1/M) Σ (ln Δx - μ̂)²
ldx = log.(db_bus.dx[db_bus.dx_pos])
M   = length(ldx)

μ_hat = mean(ldx)
σ_hat = sqrt(mean((ldx .- μ_hat) .^ 2))

# O(Y | θ̂) = Σ [ -(ln Δx - μ)²/(2σ²) - ln Δx - ln σ - ½ ln 2π ]
O_hat = sum(-(ldx .- μ_hat) .^ 2 ./ (2σ_hat^2) .- ldx .- log(σ_hat) .- 0.5log(2π))

# Checking if calculated log likelihood matches lognormal with the same parameters
O_check = sum(logpdf.(LogNormal(μ_hat, σ_hat), exp.(ldx)))

#### Replacement hazard ####

# FOC per bin: D_n/h_n - (K_n - D_n)/(1 - h_n) = 0  ⇒  ĥ_n = D_n / K_n
df_bins.h_hat = ifelse.(df_bins.K .> 0, df_bins.D ./ max.(df_bins.K, 1), NaN)

# R(Y | θ̂) = Σ_n [ D_n ln ĥ_n + (K_n - D_n) ln(1 - ĥ_n) ], with 0·ln 0 = 0
R_hat = sum(
  (df_bins.D[n] > 0 ? df_bins.D[n] * log(df_bins.h_hat[n]) : 0.0) +
  (df_bins.K[n] - df_bins.D[n] > 0 ? (df_bins.K[n] - df_bins.D[n]) * log(1 - df_bins.h_hat[n]) : 0.0)
  for n in 1:N_BINS if df_bins.K[n] > 0
)

L_hat = O_hat + R_hat

#### Report ####

@printf("μ̂ = %.4f | σ̂ = %.4f\n", μ_hat, σ_hat)
@printf("O(Y|θ̂) = %.2f  | R(Y|θ̂) = %.2f | L(Y|θ̂) = %.2f\n", O_hat, R_hat, L_hat)
df_bins[:, [:bin, :x_lo, :x_hi, :K, :D, :h_hat]]

# Hazard plot on bin edges: step holds ĥ_n over [x̄_{n-1}, x̄_n) (empty bins would show as gaps)
x_step = vcat(df_bins.x_lo, df_bins.x_hi[end]) ./ 1000
h_step = vcat(df_bins.h_hat, df_bins.h_hat[end])
p_haz = plot(x_step, h_step, seriestype = :steppost, legend = false,
             xlabel = "Mileage since last replacement (000s)", ylabel = "ĥ(x)",
             title = "Binned replacement hazard, N = $(N_BINS)")
savefig(p_haz, "output/hazard_binned.png")


#####################################################
#### Q2: Bernstein hazard, h(x | f), n = 5 ####
#####################################################

# h(x | f) = Σ_{k=1}^n f_k C(n,k) (x/x̄)^k (1 - x/x̄)^{n-k},  f ∈ [0,1]^n, h(0) = 0 by dropping k = 0.
# h is linear in f: h = B f. Then R(f) = Σ d ln(Bf) + (1-d) ln(1-Bf) is concave in f:
#   ∇R = Bᵀ w,          w_i = d_i/h_i - (1-d_i)/(1-h_i)
#   ∇²R = -Bᵀ diag(v) B, v_i = d_i/h_i² + (1-d_i)/(1-h_i)²
# So any interior max is unique; differences across starts are about the algorithms, not the surface.
# O(Y | θ) is unchanged, so (μ̂, σ̂) carry over from Q1 and only f is optimized.

const N_BERN = 5
const X_BAR  = X_MAX - 1    # x̄ = max(x_it)

# Basis matrix (rows = obs, cols = k = 1..n)
u_bern = db_bus.x_state ./ X_BAR
B = hcat([binomial(N_BERN, k) .* u_bern .^ k .* (1 .- u_bern) .^ (N_BERN - k) for k in 1:N_BERN]...)
d_vec = Float64.(db_bus.d)

# Replacement log likelihood; -Inf outside the parameter space so both optimizers reject those points
function R_bern(f)
  (any(f .< 0) || any(f .> 1)) && return -Inf
  h = B * f
  (any(h[d_vec .== 1] .<= 0) || any(h[d_vec .== 0] .>= 1)) && return -Inf
  sum(ifelse.(d_vec .== 1, log.(max.(h, eps())), log.(max.(1 .- h, eps()))))
end

# Branch on d: obs at x = 0 have h = 0 exactly, and d/h with d = 0 would give 0/0
function grad_hess_bern(f)
  h = B * f
  w = ifelse.(d_vec .== 1, 1 ./ h, -1 ./ (1 .- h))
  v = ifelse.(d_vec .== 1, 1 ./ h .^ 2, 1 ./ (1 .- h) .^ 2)
  B' * w, -B' * (v .* B)
end

#### Newton-Raphson ####

function newton_raphson(fobj, gh, x0; box = false, tol = 1e-8, dec_tol = 1e-10, maxit = 500, maxhalf = 60)
  x, Fx = copy(x0), fobj(x0)
  proj = box ? (z -> clamp.(z, 0.0, 1.0)) : identity
  for it in 1:maxit
    g, H = gh(x)
    free = box ? .!(((x .<= 0) .& (g .< 0)) .| ((x .>= 1) .& (g .> 0))) : trues(length(x))
    maximum(abs, g[free]) < tol && return (x = x, F = Fx, iter = it - 1, converged = true)
    dir = zeros(length(x))
    dir[free] = H[free, free] \ g[free]
    -0.5 * dot(g[free], dir[free]) < dec_tol && return (x = x, F = Fx, iter = it - 1, converged = true)
    λ, nhalf = 1.0, 0
    xn = proj(x .- λ .* dir)
    Fn = fobj(xn)
    while !(Fn > Fx)
      λ /= 2
      nhalf += 1
      nhalf > maxhalf && return (x = x, F = Fx, iter = it, converged = false)
      xn = proj(x .- λ .* dir)
      Fn = fobj(xn)
    end
    x, Fx = xn, Fn
  end
  (x = x, F = Fx, iter = maxit, converged = false)
end

#### Nelder-Mead ####

function nelder_mead(fobj, x0; step = 0.1, α = 1.0, γ = 1.0, μ = 0.5, τ = 0.5, tol = 1e-10, maxit = 20_000)
  N = length(x0)
  X = [copy(x0) for _ in 0:N]
  for n in 1:N
    X[n + 1][n] += step * (x0[n] == 0 ? 1.0 : x0[n])
  end
  F = fobj.(X)
  for it in 1:maxit
    p = sortperm(F, rev = true)
    X, F = X[p], F[p]
    abs(F[1] - F[end]) < tol && return (x = X[1], F = F[1], iter = it - 1, converged = true)
    c  = sum(X[1:N]) ./ N                     # centroid without worst point
    xR = c .+ α .* (c .- X[end])
    fR = fobj(xR)
    if fR > F[1]                              # expansion
      xE = xR .+ γ .* (xR .- c)
      fE = fobj(xE)
      fE > fR && ((xR, fR) = (xE, fE))
    end
    if fR > F[N]                              # improvement on second-worst: accept
      X[end], F[end] = xR, fR
      continue
    end
    xhat = fR > F[end] ? xR : X[end]          # contraction
    xC = c .+ μ .* (xhat .- c)
    fC = fobj(xC)
    if fC > F[end]
      X[end], F[end] = xC, fC
      continue
    end
    for n in 2:(N + 1)                        # shrink toward best
      X[n] = X[n] .+ τ .* (X[1] .- X[n])
      F[n] = fobj(X[n])
    end
  end
  p = sortperm(F, rev = true)
  (x = X[p[1]], F = F[p[1]], iter = maxit, converged = false)
end

# Restart NM from its own best point until a restart gains < tol
function nelder_mead_restart(fobj, x0; restarts = 50, tol = 1e-8, kw...)
  out = nelder_mead(fobj, x0; kw...)
  iters = out.iter
  for r in 1:restarts
    new = nelder_mead(fobj, out.x; kw...)
    iters += new.iter
    gain = new.F - out.F
    out = new
    gain < tol && return (x = out.x, F = out.F, iter = iters, converged = true)
  end
  (x = out.x, F = out.F, iter = iters, converged = false)
end

#### Starting values ####

# Flat starts across the box, two shaped starts, and 15 uniform draws
Random.seed!(880)
starts = vcat(
  [fill(c, N_BERN) for c in [1e-4, 0.01, 0.05, 0.2, 0.5, 0.9, 0.999]],
  [collect(range(0.001, 0.3, length = N_BERN)), collect(range(0.3, 0.001, length = N_BERN))],
  [rand(N_BERN) for _ in 1:15]
)
start_lbl = vcat(["flat $(c)" for c in [1e-4, 0.01, 0.05, 0.2, 0.5, 0.9, 0.999]],
                 ["increasing", "decreasing"],
                 ["rand $(i)" for i in 1:15])

# Run one method over all starts; collect into a table
function run_starts(method, label)
  rows = map(eachindex(starts)) do s
    t = @elapsed out = method(starts[s])
    (method = label, start = start_lbl[s], R0 = R_bern(starts[s]), R = out.F,
     iter = out.iter, converged = out.converged, secs = t,
     f1 = out.x[1], f2 = out.x[2], f3 = out.x[3], f4 = out.x[4], f5 = out.x[5])
  end
  DataFrame(rows)
end

#### (a) Newton-Raphson ####

# Plain NR
df_nr = run_starts(x0 -> newton_raphson(R_bern, grad_hess_bern, x0), "NR")
df_nr

# Projected NR with active set
df_nrb = run_starts(x0 -> newton_raphson(R_bern, grad_hess_bern, x0; box = true), "NR-box")
df_nrb

#### (b) Nelder-Mead ####

# Plain NM: simplex collapses against the -Inf wall at the bound
df_nm = run_starts(x0 -> nelder_mead(R_bern, x0), "NM")
df_nm

# NM with restarts
df_nmr = run_starts(x0 -> nelder_mead_restart(R_bern, x0), "NM-restart")
df_nmr

# Cross-check own NM against Optim.jl (minimizes -R)
optim_nm = optimize(f -> -R_bern(f), fill(0.05, N_BERN), NelderMead(),
                    Optim.Options(iterations = 20_000, g_tol = 1e-12))
@printf("Optim NM: R = %.6f, f = %s\n", -Optim.minimum(optim_nm), string(round.(Optim.minimizer(optim_nm), digits = 5)))

#### (c) Hybrid ####

const K_NM = 50

function hybrid(x0; box = false)
  nm = nelder_mead(R_bern, x0; maxit = K_NM)
  nr = newton_raphson(R_bern, grad_hess_bern, nm.x; box = box)
  (x = nr.x, F = nr.F, iter = nm.iter + nr.iter, converged = nr.converged)
end

df_hy = run_starts(hybrid, "NM($(K_NM)) → NR")
df_hy

df_hyb = run_starts(x0 -> hybrid(x0; box = true), "NM($(K_NM)) → NR-box")
df_hyb

#### (d) Summary ####

df_q2 = vcat(df_nr, df_nrb, df_nm, df_nmr, df_hy, df_hyb)

# Reference optimum from already implemented solver

m_ref = Model(Ipopt.Optimizer)
set_silent(m_ref)
@variable(m_ref, 0 <= f_ref[1:N_BERN] <= 1, start = 0.05)
@expression(m_ref, h_ref, B * f_ref)
@objective(m_ref, Max, sum(log(h_ref[i]) for i in findall(d_vec .== 1)) +
                       sum(log(1 - h_ref[i]) for i in findall(d_vec .== 0)))
optimize!(m_ref)
f_ipopt = clamp.(value.(f_ref), 0.0, 1.0)
R_star  = max(R_bern(f_ipopt), maximum(df_q2.R))
@printf("Ipopt reference: R = %.6f, f = %s\n", R_bern(f_ipopt), string(round.(f_ipopt, digits = 5)))

df_q2.gap = R_star .- df_q2.R
df_q2.hit = df_q2.gap .< 1e-4

df_summary = combine(groupby(df_q2, :method),
  :hit       => mean   => :share_at_opt,
  :converged => mean   => :share_converged,
  :gap       => (g -> maximum(g[isfinite.(g)])) => :worst_gap,
  :iter      => median => :median_iter,
  :secs      => median => :median_secs)
df_summary

f_hat = Vector(df_q2[argmax(df_q2.R), [:f1, :f2, :f3, :f4, :f5]])
L_bern = O_hat + R_star
@printf("Bernstein (n = %d): f̂ = %s\n", N_BERN, string(round.(f_hat, digits = 5)))
@printf("R(f̂) = %.4f vs binned R = %.4f | L(Y|θ̂) = %.4f vs binned %.4f\n", R_star, R_hat, L_bern, L_hat)

# Fitted Bernstein hazard over the binned estimate
x_plot = range(0, X_BAR, length = 400)
u_plot = x_plot ./ X_BAR
h_plot = sum(f_hat[k] .* binomial(N_BERN, k) .* u_plot .^ k .* (1 .- u_plot) .^ (N_BERN - k) for k in 1:N_BERN)
p_bern = plot(x_step, h_step, seriestype = :steppost, label = "Binned (N = $(N_BINS))",
              xlabel = "Mileage since last replacement (000s)", ylabel = "h(x)", legend = :topleft)
plot!(p_bern, x_plot ./ 1000, h_plot, label = "Bernstein (n = $(N_BERN))", lw = 2)
savefig(p_bern, "output/hazard_bern.png")
