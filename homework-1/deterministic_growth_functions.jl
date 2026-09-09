#keyword-enabled structure to hold model primitives
@kwdef struct Primitives
    # non-stochastic primitives
    β::Float64 = 0.99 #discount rate
    δ::Float64 = 0.025 #depreciation rate
    α::Float64 = 0.36 #capital share
    k_min::Float64 = 0.01 #capital lower bound
    k_max::Float64 = 90.0 #capital upper bound
    nk::Int64 = 1000 #number of capital grid points
    k_grid::Array{Float64,1} = collect(range(start=k_min, stop=k_max, length=nk)) #capital grid
    # stochastic primitives 
    z_grid::Array{Float64, 1} = [1.25, 0.2] # support of policy shocks 
    Π::Array{Float64, 2} = [0.977 0.023; 0.074 0.926] # transition matrix
    nz::Int64 = length(z_grid) # cardinality of technology support
end

#structure that holds model results
mutable struct Results
    val_func::Array{Float64, 2} #value function
    pol_func::Array{Float64, 2} #policy function
end

#function for initializing model primitives and results
function Initialize()
    prim = Primitives() #initialize primtiives
    val_func = zeros(prim.nk) #initial value function guess
    pol_func = zeros(prim.nk) #initial policy function guess
    res = Results(val_func, pol_func) #initialize results struct
    prim, res #return deliverables
end

#Bellman Operator
function Bellman(prim::Primitives,res::Results)

    # upack primitives and current results
    (; val_func, pol_func) = res 
    (; β, δ, α, k_min, k_max, nk, k_grid, z_grid, Π, nz) = prim

    # set placeholder for next value and policy function guesses 
    v_next = zero(val_func) 
    g_next = zero(pol_func)

    # init choice index, used for exploiting 
    # monotonicity of the policy function 
    choice_lower = 1 

    # loop over the support of policy shocks 
    for z_ix in eachindex(z_grid)

        # compute expected continutation value 
        EV = val_func * Π[z_ix, :]

        # conditional on a policy shock, loop over capital grid 
        for (k_ix, k) in enumerate(k_grid)

            # init candidate maximum 
            candidate_max = -Inf 

            # set budget constraint with current capital
            budget = k^α + (1 - δ) * k

            # loop over possible selection of k' 
            for kp_ix in choice_lower:nk 

                # calculate consumption given k'
                c = budget - k_grid[kp_ix] 

                # if feasible, check induced indirect utility
                if c>0 

                    val = log(c) + β * EV[kp_ix, z_ix]

                    # update value and policy function
                    if val > candidate_max 
                        candidate_max = val 
                        g_next[k_ix, z_ix] = k_grid[kp_ix] 
                        choice_lower = kp_index
                    end
                end
            end
            v_next[k_ix, z_ix] = candidate_max #update value function
        end 
    end 

    return v_next, g_next
end

#Value function iteration
function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)
    n = 0 #counter

    while err>tol #begin iteration
        v_next = Bellman(prim, res) #spit out new vectors
        err = maximum(abs.(v_next.-res.val_func)) #/abs(v_next[prim.nk, 1]) #reset error level
        res.val_func = v_next #update value function
        n+=1
    end
    println("Value function converged in ", n, " iterations.")
end

#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res) #in this case, all we have to do is the value function iteration!
end
##############################################################################
