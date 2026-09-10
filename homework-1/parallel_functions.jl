#= 
This code is a parallelized version of the VFI code for the neoclassical growth model.
The main difference is that the Bellman operator is parallelized using the @distributed macro.
September 2024
=#


@everywhere @kwdef struct Primitives
    # non-stochastic primitives 
    β::Float64 = 0.99
    δ::Float64 = 0.025
    α::Float64 = 0.36
    k_min::Float64 = 0.01
    k_max::Float64 = 90.0
    nk::Int64 = 1000
    k_grid::SharedArray{Float64} = collect(range(start=k_min, stop=k_max, length=nk))
    # stochastic primitives 
    z_grid::SharedArray{Float64} = [1.25, 0.2]
    Π::SharedArray{Float64, 2} = [0.977 0.023; 0.074 0.926]
    nz::Int64 = length(z_grid)
end


@everywhere @kwdef mutable struct Results
    val_func::SharedArray{Float64, 2}
    pol_func::SharedArray{Float64, 2}
end


@everywhere function Initialize()
    prim = Primitives()
    val_func = SharedArray{Float64, 2}(zeros(prim.nk, prim.nz))
    pol_func = SharedArray{Float64, 2}(zeros(prim.nk, prim.nz))
    res = Results(val_func, pol_func)
    return prim, res
end


function Bellman(prim::Primitives, res::Results)

    # upack primitives and current results
    (; val_func, pol_func) = res 
    (; β, δ, α, nk, k_grid, z_grid, Π) = prim
    
    v_next = SharedArray{Float64, 2}(zero(val_func))
    g_next = SharedArray{Float64, 2}(zero(pol_func))

    # NOTE: I am choosing to parallelize over the 
    # capital grid, since it is large while the 
    # support of the shocks has only two values 
    #                               - Aaron 

    for z_ix in eachindex(z_grid)

        EV = val_func * Π[z_ix, :]

        @sync @distributed for k_ix in 1:nk

            k = k_grid[k_ix]
            z = z_gridp[z_ix]

            candidate_max = -Inf
            budget = z * k^α + (1 - δ) * k
            
            for kp_ix in 1:nk

                c = budget - k_grid[kp_ix]
                
                if c > 0

                    val = log(c) + β * EV[kp_ix]
                    
                    if val > candidate_max
                        candidate_max = val
                        g_next[k_ix, z_ix] = k_grid[kp_ix] 
                    end
                end
            end
            v_next[k_ix, z_ix] = candidate_max
        end
    end 


    return v_next
end


function V_iterate(prim::Primitives, res::Results; tol::Float64 = 1e-6, err::Float64 = 100.0)

    n = 0

    while err > tol
        v_next, g_next = Bellman(prim, res)
        err = maximum(abs.(v_next .- res.val_func))
        res.val_func .= v_next
        res.pol_func .= g_next
        n += 1
    end
    println("Value function converged in ", n, " iterations.")
end


#solve the model
function Solve_model(prim::Primitives, res::Results)
    V_iterate(prim, res)
end