
# -- hugget algorithm -- #
    
    # take θ as given 

    # Outer loop: Bisection over q in [0,1]
        
        # Inner loop 1: Solve stochastic VFI
            # return V and g
        
        # Inner loop 2: given g, solve for invarient wealth dist μ
            # return μ

    # check if ∫g(s,a;q)μ dads = 0

    # -- end of algorithm -- #      

# define model primitives 
@kwdef struct Primitives
    
    # non-stochastic primitives
    β::Float64 = 0.9932 # discount rate
    α::Float64 = 1.5 # coeff of relative risk aversion
    a_min::Float64 = -2.0 # asset grid lower bound 
    a_max::Float64 = 5.0 # asset grid upper bound 
    n_a::Int64 = 1000 # number of grid points 
    a_grid = collect(
        range(start=a_min, stop=a_max, length=n_a)
    )

    # stochastic primitive 
    s_grid::Array{Float64, 1} = [1.0, 0.5] # income shocks
    n_s::Int64 = length(s_grid) # cardinality of shock support
    Π::Array{Float64, 2} = [0.97 0.03; 0.5 0.5] # transition matrix 

end

# define endogenous objects 
# NOTE: instead of defining a policy function, 
# I am keeping track of the indicies in the asset 
# grid where the value of the optimal asset choice 
# is located. This helps in the inner_loop_2 step. 
# I will still refer to it as policy "function"
mutable struct Results
    val_func::Array{Float64, 2} # value function
    pol_func::Array{Int64, 2} # policy function
    sta_dist::Array{Float64, 2} # stationary dist
    q::Float64 # bond price 
end

function initialize()
    """Initialize model (Primitives, Results)"""

    # initialize primitives 
    prim = Primitives()

    # value and policy function placeholder 
    val_func = zeros(prim.n_a, prim.n_s)
    pol_func = zeros(Int64, prim.n_a, prim.n_s)

    # stationary distribution placeholder
    sta_dist = zeros(prim.n_a, prim.n_s)

    # set initial bond price 
    q = 0.0

    # package placeholders into Results
    res = Results(val_func, pol_func, sta_dist, q)

    return prim, res 
end

# define momentary utility function 
u(c::Float64, α::Float64) = (c^(1- α) - 1) / (1 - α)

function bellman(res::Results, prim::Primitives)
    """Bellman operator to update guess of (val_func, pol_func)"""

    # unpack primitives and current results
    (; val_func, pol_func, q) = res 
    (; β, α, n_a, a_grid, s_grid, Π) = prim 

    # placeholder for next guess of V 
    v_next = zero(val_func)

    # loop over support of income shocks 
    for (s_ix, s) in enumerate(s_grid)

        EV = val_func * Π[s_ix, :]

        # init choice index, used for exploiting 
        # monotonicity of the policy function 
        choice_lower = 1 

        # conditional on income shock, loop over capital grid
        for (a_ix, a) in enumerate(a_grid)

            # init candidate maximum 
            candidate_max = -Inf

            # init candidate ix for optimal a' 
            best_ix = choice_lower

            # calculate current income 
            income = s + a 

            # loop over possible choice of a' 
            for ap_ix in choice_lower:n_a

                # calculate consumption as residual of asset choice 
                c = income - q * a_grid[ap_ix]

                # since c decreases in a', all
                # larger a' are also infeasible 
                if c <= 0
                    break
                end
                
                # given feasible c, compute indirect utility 
                val = u(c, α) + β * EV[ap_ix]

                # update value and optimal policy index 
                if val > candidate_max
                    candidate_max = val
                    best_ix = ap_ix 
                
                # if V starts decreasing, we have already found
                # the max since V is concave 
                elseif val < candidate_max
                    break 
                end 
            end 
            
            # update value and policy function
            v_next[a_ix, s_ix] = candidate_max
            pol_func[a_ix, s_ix] = best_ix

            # note that g(a, s) is increasing in a 
            choice_lower = best_ix
        end 
    end 

    return v_next, pol_func
end 

function kolmogorov(res::Results, prim::Primitives)
    """kolmogorov-forward operator to update guess of sta_dist"""

    # unpack primitives and current results
    (; pol_func, sta_dist) = res
    (; n_a, n_s, Π) = prim

    # placeholder for next guess of stationary distribution
    dist_next = zero(sta_dist)

    # loop over current income shock
    for s_ix in 1:n_s

        # loop over current asset holdings
        for a_ix in 1:n_a

            # find optimal next-period asset index
            ap_ix = pol_func[a_ix, s_ix]

            # loop over next-period income shock
            for sp_ix in 1:n_s

                # move current probability mass to next-period state
                dist_next[ap_ix, sp_ix] +=
                    sta_dist[a_ix, s_ix] * Π[s_ix, sp_ix]

            end
        end
    end

    return dist_next
end

function inner_loop_1!(
    res::Results,
    prim::Primitives;
    tol::Float64 = 1e-6,
    err::Float64 = Inf
)
    """Solve inner loop #1 for (val_func, pol_func)"""

    while err > tol

        # apply Bellman operator
        v_next, pol_func = bellman(res, prim)

        # compute convergence error
        err = maximum(abs.(v_next .- res.val_func))

        # update value and policy functions
        res.val_func .= v_next
        res.pol_func .= pol_func
    end

    return nothing
end


function inner_loop_2!(
    res::Results,
    prim::Primitives;
    tol::Float64 = 1e-6,
    err::Float64 = Inf
)
    """Solve inner loop #2 for sta_dist"""

    # initialize distribution if necessary
    if sum(res.sta_dist) == 0.0
        # initial guess with uniform weights
        res.sta_dist .= 1.0 / (prim.n_a * prim.n_s)
    end

    while err > tol

        # apply Kolmogorov-forward operator
        dist_next = kolmogorov(res, prim)

        # compute convergence error
        err = maximum(abs.(dist_next .- res.sta_dist))

        # update stationary distribution
        res.sta_dist .= dist_next
    end

    return nothing
end

function solve_GE(
    prim::Primitives,
    res::Results;
    q_lower::Float64 = 0.0,
    q_upper::Float64 = 1.0,
    tol::Float64 = 1e-8
)
    """Solve for equilibrium using bisection"""

    while q_upper - q_lower > tol

        # update guess of bond price
        res.q = (q_lower + q_upper) / 2

        # check if HH problem is feasible over entire state space
        c_max_min = minimum(prim.s_grid) +
                    prim.a_min -
                    res.q * prim.a_min

        if c_max_min <= 0

            # candidate q does not yield a feasible choice
            # for every (a, s) in A × S
            q_lower = res.q
            continue
        end

        # solve HH problem
        inner_loop_1!(res, prim)

        # solve for stationary distribution
        inner_loop_2!(res, prim)

        # calculate aggregate asset demand
        asset_demand = 0.0

        for s_ix in 1:prim.n_s
            for a_ix in 1:prim.n_a

                # recover optimal asset choice
                ap_ix = res.pol_func[a_ix, s_ix]
                ap = prim.a_grid[ap_ix]

                # weight asset choice by stationary probability
                asset_demand +=
                    ap * res.sta_dist[a_ix, s_ix]
            end
        end

        # update bond-price bounds
        if asset_demand > 0
            q_lower = res.q
        else
            q_upper = res.q
        end
    end

    # set equilibrium price to midpoint of final bracket
    res.q = (q_lower + q_upper) / 2

    # solve model at reported equilibrium price
    inner_loop_1!(res, prim)
    inner_loop_2!(res, prim)

    return nothing
end

