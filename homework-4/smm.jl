using Random, Statistics, LinearAlgebra, Optim, Distributions

@kwdef struct Primitives
    b_0::Array{Float64, 1} = [0.5, 1.0]
    x_0::Float64 = 0.0
    T::Int64 = 200
end

mutable struct Data 
    series::Array{Float64, 1} 
    errors::Array{Float64, 2}
end 

function true_DGP(prim::Primitives; rng=Random.default_rng())
    """generate the true data of length T"""

    # unpack parameters
    ρ, σ = prim.b_0

    # generate innovations
    errors = σ .* randn(rng, prim.T)

    # generate AR(1) process
    x = zeros(prim.T)
    x_prev = prim.x_0
    for ix in eachindex(x)
        x[ix] = ρ * x_prev + errors[ix]
        x_prev = x[ix]
    end

    return x 
end

function simulate_errs(H::Int64, prim::Primitives; rng=Random.default_rng())
    """generate a T x H matrix of errors for simulation"""
    return randn(rng, prim.T, H)
end 

function initialize(H::Int64; rng=Random.default_rng())
    """collect data and primitives given H"""

    # initialize primitives 
    prim = Primitives()

    # generate true data 
    series = true_DGP(prim, rng=rng)

    # generate simulate errors 
    errors = simulate_errs(H, prim, rng=rng)

    return Data(series, errors)
end

function simulate(
    b::Array{Float64},
    prim::Primitives,
    errors::Array{Float64, 2}
    )
    """generate T x H simulated series given parameters and fixed errors"""

    # unpack params and series dims 
    ρ, σ = b
    T, H = size(errors)

    # simulate H series of length T 
    y = zeros(T, H)
    for h in 1:H
        y_prev = prim.x_0
        for t in 1:T
            y[t, h] = ρ * y_prev + σ * errors[t, h]
            y_prev = y[t, h]
        end
    end

    return y
end

function moments(
    series::AbstractVector{Float64},
    selected::Tuple{Vararg{Symbol}}
    )
    """construct selected vector of moments"""

    # get length of time series
    T = length(series)

    # compute mean
    μ = mean(series)

    # compute variance 
    deviations = series .- μ
    variance = sum(abs2, deviations) / T

    # compute autocovariance
    autocov = sum(
        deviations[t] * deviations[t-1]
        for t in 2:T
    ) / T

    # contruct moments given selected
    m = zeros(length(selected))
    for (i, moment) in enumerate(selected)
        if moment == :mean
            m[i] = μ
        elseif moment == :variance
            m[i] = variance
        elseif moment == :autocovariance
            m[i] = autocov
        elseif moment == :autocorrelation
            m[i] = autocov / variance
        else
            error("Unknown moment: $moment")
        end
    end

    return m
end

function simulated_moments(
    b::Array{Float64},
    prim::Primitives,
    errors::Array{Float64, 2},
    selected::Tuple{Vararg{Symbol}}
    )
    """Compute average moments across H simulated series"""

    # given guess of b, simualte H series 
    y = simulate(b, prim, errors)

    # get number of simulation
    H = size(y, 2)

    # construct simulated moments given selected 
    m_sim = zeros(length(selected))
    for h in 1:H
        m_sim .+= moments(view(y, :, h), selected)
    end

    return m_sim ./ H
end

function smm_objective(
    b::Array{Float64},
    prim::Primitives,
    data::Data,
    m_data::Array{Float64},
    selected::Tuple{Vararg{Symbol}},
    W::Array{Float64}
    )
     """compute value of SMM objective given W"""

    # unpack params 
    ρ, σ = b
     
    # enforce stationary and positive standard deviation
    if !(abs(ρ) < 1 && σ > 0)
        return Inf
    end

    # compute simulated moments
    m_sim = simulated_moments(b, prim, data.errors, selected)

    # Compute moment errors
    g = m_data - m_sim

    # Compute objective
    J = dot(g, W * g)

    return J
end

function moment_contributions(
    series::AbstractVector{Float64},
    selected::Tuple{Vararg{Symbol}}
    )
    """construct T x K matrix of per-period moment contributions"""

    # get length of time series
    T = length(series)

    # compute mean and deviations
    μ = mean(series)
    d = series .- μ

    # compute variance
    v = sum(abs2, d) / T

    # initialize matrix of moment contributions
    C = zeros(T, length(selected))

    # construct contributions given selected
    for (j, moment) in enumerate(selected)
        if moment == :mean
            C[:, j] .= series

        elseif moment == :variance
            C[:, j] .= d .^ 2

        elseif moment == :autocovariance
            C[2:T, j] .= d[2:T] .* d[1:T-1]

        elseif moment == :autocorrelation
            # compute autocovariance and autocorrelation
            γ = sum(d[2:T] .* d[1:T-1]) / T
            r = γ / v

            # construct linearized autocorrelation contributions
            C[:, j] .= r .- (r / v) .* d .^ 2
            C[2:T, j] .+= (d[2:T] .* d[1:T-1]) ./ v

        else
            error("Unknown moment: $moment")
        end
    end

    return C
end

function newey_west(
    y::AbstractMatrix{Float64},
    selected::Tuple{Vararg{Symbol}};
    lags::Int64=4
    )
    """estimate long-run covariance matrix using Newey-West across H simulations"""

    # unpack series dimensions and number of moments
    T, H = size(y)
    n = length(selected)

    # ensure valid number of lags
    @assert 0 <= lags < T

    # construct moment contributions for each simulated series
    C = [
        moment_contributions(view(y, :, h), selected)
        for h in 1:H
    ]

    # compute average moments across simulations
    m_bar = zeros(n)
    for h in 1:H
        m_bar .+= vec(sum(C[h], dims=1))
    end
    m_bar ./= T * H

    # center moment contributions
    for h in 1:H
        C[h] .-= reshape(m_bar, 1, n)
    end

    # initialize long-run covariance matrix
    S = zeros(n, n)

    # compute Newey-West covariance across lags
    for lag in 0:lags

        # compute lagged covariance across simulations
        Γ = zeros(n, n)
        for h in 1:H
            Γ .+= C[h][lag+1:T, :]' * C[h][1:T-lag, :]
        end
        Γ ./= T * H

        # apply Bartlett weights to lagged covariances
        if lag == 0
            S .+= Γ
        else
            weight = 1 - lag / (lags + 1)
            S .+= weight .* (Γ + Γ')
        end

    end

    # adjust for simulation variance
    S .*= (1 + 1/H)

    return Symmetric(S)
end

function estimate_smm(
    prim::Primitives,
    data::Data,
    selected::Tuple{Vararg{Symbol}},
    b_init::AbstractVector{Float64};
    lags::Int64=4
    )
    """two-stage SMM estimation using optimal weighting matrix"""

    # compute empirical moments
    m_data = moments(data.series, selected)
    n = length(m_data)

    # initialize first-stage weighting matrix
    W1 = Matrix{Float64}(I, n, n)

    # construct first-stage objective
    obj1 = b -> smm_objective(
        b, prim, data, m_data, selected, W1
    )

    # estimate first-stage parameters
    result1 = optimize(obj1, copy(b_init), NelderMead())
    b1 = Optim.minimizer(result1)

    # simulate series using first-stage estimates
    y1 = simulate(b1, prim, data.errors)

    # estimate long-run covariance matrix
    S = newey_west(y1, selected; lags=lags)

    # construct optimal weighting matrix
    W2 = S \ W1

    # construct second-stage objective
    obj2 = b -> smm_objective(
        b, prim, data, m_data, selected, W2
    )

    # estimate second-stage parameters
    result2 = optimize(obj2, copy(b1), NelderMead())
    b2 = Optim.minimizer(result2)

    # return estimates and estimation results
    return (
        b1 = b1,
        b2 = b2,
        m_data = m_data,
        S = S,
        W1 = W1,
        W2 = W2,
        J1 = Optim.minimum(result1),
        J2 = Optim.minimum(result2),
        converged1 = Optim.converged(result1),
        converged2 = Optim.converged(result2)
    )
end

function jacobian(
    g::Function,
    b::AbstractVector{Float64};
    step::Float64=1e-5
)
    """approximate jacobian of moment function"""

    # evaluate moment function at b
    g_b = g(b)

    # get dimensions of jacobian
    n = length(g_b)
    l = length(b)

    # initialize jacobian
    G = zeros(n, l)

    # approximate derivatives for each parameter
    for j in 1:l

        # perturb parameter j
        b_minus = copy(b)
        b_minus[j] -= step

        # compute backward finite difference
        G[:, j] .= (g_b - g(b_minus)) ./ step
    end

    return G
end

function ags_statistic(
    G::AbstractMatrix{Float64},
    W::AbstractMatrix{Float64}
)
    """compute AGS sensitivity matrix"""

    # compute weighted jacobian products
    A = G' * W * G
    B = G' * W

    # compute sensitivity matrix
    Λ = -(A \ B)

    return Λ
end

function j_test(
    J::Float64,
    T::Int64,
    H::Int64,
    n::Int64,
    l::Int64;
    adjusted_W::Bool=true
)
    """compute value of J-test"""

    # compute degrees of freedom
    df = n - l

    # compute J statistic
    if adjusted_W
        J_stat = T * J
    else
        J_stat = (T * H / (H + 1)) * J
    end

    # compute p-value for overidentified model
    if df > 0
        pvalue = ccdf(Chisq(df), J_stat)
    else
        pvalue = missing
    end

    return (
        statistic = J_stat,
        df = df,
        pvalue = pvalue
    )
end

function smm_inference(
    prim::Primitives,
    data::Data,
    selected::Tuple{Vararg{Symbol}},
    results;
    step::Float64=1e-5
)
    """compute SMM standard errors and diagnostic statistics"""

    # unpack second-stage estimates and weighting matrix
    b2 = results.b2
    W2 = results.W2

    # construct moment error function
    g = b -> results.m_data - simulated_moments(
        b, prim, data.errors, selected
    )

    # approximate jacobian at second-stage estimates
    G = jacobian(g, b2; step=step)

    # compute parameter covariance matrix
    l = length(b2)
    A = G' * W2 * G

    V = (A \ Matrix{Float64}(I, l, l)) / prim.T

    # compute standard errors
    se = sqrt.(diag(V))

    # compute AGS sensitivity statistics
    Λ = ags_statistic(G, W2)

    # compute J-test
    H = size(data.errors, 2)
    n = length(selected)

    J = j_test(results.J2, prim.T, H, n, l)

    # compute singular values of jacobian
    singular_values = svdvals(G)

    # return inference results
    return (
        G = G,
        V = V,
        se = se,
        Λ = Λ,
        J = J,
        singular_values = singular_values
    )
end

function bootstrap_smm(
    prim::Primitives,
    H::Int64,
    selected::Tuple{Vararg{Symbol}},
    b_init::AbstractVector{Float64};
    B::Int64=500,
    lags::Int64=4,
    rng=Random.default_rng()
)
    """repeat SMM estimation across independently simulated datasets"""

    # get number of parameters
    l = length(b_init)

    # initialize matrices to store estimates
    b1_boot = zeros(l, B)
    b2_boot = zeros(l, B)

    # initialize convergence indicators
    converged = falses(B)

    # run bootstrap replications
    for r in 1:B

        # generate new true data
        series = true_DGP(prim; rng=rng)

        # generate new simulated innovations
        errors = simulate_errs(H, prim; rng=rng)

        # construct data for replication
        data_r = Data(series, errors)

        # estimate two-stage SMM
        results = estimate_smm(
            prim, data_r, selected, b_init; lags=lags
        )

        # store parameter estimates
        b1_boot[:, r] .= results.b1
        b2_boot[:, r] .= results.b2

        # record convergence
        converged[r] = results.converged1 && results.converged2
    end

    # compute means of parameter estimates
    mean_b1 = vec(mean(b1_boot, dims=2))
    mean_b2 = vec(mean(b2_boot, dims=2))

    # compute standard deviations of parameter estimates
    sd_b1 = vec(std(b1_boot, dims=2))
    sd_b2 = vec(std(b2_boot, dims=2))

    return (
        b1 = b1_boot,
        b2 = b2_boot,
        mean_b1 = mean_b1,
        mean_b2 = mean_b2,
        sd_b1 = sd_b1,
        sd_b2 = sd_b2,
        converged = converged
    )
end

function smm_surface(
    prim::Primitives,
    data::Data,
    selected::Tuple{Vararg{Symbol}},
    ρ_grid::AbstractVector{Float64},
    σ_grid::AbstractVector{Float64}
)
    """evaluate first-stage SMM objective over parameter grid"""

    # compute empirical moments
    m_data = moments(data.series, selected)

    # construct identity weighting matrix
    n = length(selected)
    W = Matrix{Float64}(I, n, n)

    # initialize objective surface
    J = zeros(length(σ_grid), length(ρ_grid))

    # evaluate objective over parameter grid
    for (i, ρ) in enumerate(ρ_grid)
        for (j, σ) in enumerate(σ_grid)

            b = [ρ, σ]

            J[j, i] = smm_objective(
                b, prim, data, m_data, selected, W
            )
        end
    end

    return J
end


# ---------------------------------------
# Problem 7: Indirect Inference
# ---------------------------------------

function ma_errors(
    series::AbstractVector{Float64},
    θ::AbstractVector{Float64}
)
    """recover MA(N) prediction errors given auxiliary parameters"""

    # get series length and MA order
    T = length(series)
    N = length(θ)

    # initialize prediction errors
    errors = zeros(T)

    # recursively recover innovations
    for t in 1:T

        # initialize prediction error
        errors[t] = series[t]

        # subtract predicted component
        for j in 1:min(N, t-1)
            errors[t] -= θ[j] * errors[t-j]
        end
    end

    return errors
end


function ma_invertible(
    θ::AbstractVector{Float64}
)
    """check invertibility of auxiliary MA(N) model"""

    # get MA order
    N = length(θ)

    # construct companion matrix
    C = zeros(N, N)
    C[1, :] .= -θ

    for j in 2:N
        C[j, j-1] = 1.0
    end

    # check reciprocal roots lie inside unit circle
    return maximum(abs.(eigvals(C))) < 1.0 - 1e-8
end


function estimate_ma(
    series::AbstractVector{Float64},
    N::Int64
)
    """estimate MA(N) parameters by minimizing squared prediction errors"""

    # initialize auxiliary parameters
    θ_init = [0.5^j for j in 1:N]

    # construct conditional sum of squared errors
    function objective(θ)

        # impose MA invertibility
        if !ma_invertible(θ)
            return Inf
        end

        # recover prediction errors
        errors = ma_errors(series, θ)

        # compute mean squared prediction error
        return sum(abs2, errors) / length(series)
    end

    # estimate auxiliary parameters
    result = optimize(
        objective,
        θ_init,
        NelderMead(),
        Optim.Options(iterations=2000)
    )

    return (
        θ = Optim.minimizer(result),
        objective = Optim.minimum(result),
        converged = Optim.converged(result)
    )
end


function auxiliary_moments(
    series::AbstractVector{Float64},
    N::Int64
)
    """compute auxiliary MA(N) estimates and sample standard deviation"""

    # estimate MA(N) auxiliary model
    result = estimate_ma(series, N)

    # compute sample standard deviation
    s = std(series)

    # construct vector of auxiliary moments
    m = vcat(result.θ, s)

    return m
end


function simulated_auxiliary_moments(
    b::Array{Float64},
    prim::Primitives,
    errors::Array{Float64, 2},
    N::Int64
)
    """average auxiliary moments across H simulated series"""

    # simulate AR(1) series given structural parameters
    y = simulate(b, prim, errors)

    # get number of simulations
    H = size(y, 2)

    # initialize simulated auxiliary moments
    m_sim = zeros(N + 1)

    # estimate auxiliary model for each simulation
    for h in 1:H
        m_sim .+= auxiliary_moments(
            view(y, :, h), N
        )
    end

    # average auxiliary moments across simulations
    return m_sim ./ H
end


function indirect_objective(
    b::Array{Float64},
    prim::Primitives,
    data::Data,
    m_data::Array{Float64},
    N::Int64
)
    """compute indirect inference objective given MA(N) auxiliary model"""

    # unpack structural parameters
    ρ, σ = b

    # enforce stationarity and positive standard deviation
    if !(abs(ρ) < 1 && σ > 0)
        return Inf
    end

    # compute simulated auxiliary moments
    m_sim = simulated_auxiliary_moments(
        b, prim, data.errors, N
    )

    # compute auxiliary moment errors
    g = m_data - m_sim

    # compute indirect inference objective
    J = dot(g, g)

    return J
end


function estimate_indirect(
    prim::Primitives,
    data::Data,
    N::Int64,
    b_init::AbstractVector{Float64}
)
    """estimate AR(1) parameters using MA(N) indirect inference"""

    # compute empirical auxiliary moments once
    m_data = auxiliary_moments(data.series, N)

    # construct indirect inference objective
    obj = b -> indirect_objective(
        b, prim, data, m_data, N
    )

    # estimate structural parameters
    result = optimize(
        obj,
        copy(b_init),
        NelderMead(),
        Optim.Options(iterations=2000)
    )

    b_hat = Optim.minimizer(result)

    # compute simulated auxiliary moments at estimates
    m_sim = simulated_auxiliary_moments(
        b_hat, prim, data.errors, N
    )

    # return estimation results
    return (
        N = N,
        b = b_hat,
        m_data = m_data,
        m_sim = m_sim,
        objective = Optim.minimum(result),
        converged = Optim.converged(result)
    )
end
