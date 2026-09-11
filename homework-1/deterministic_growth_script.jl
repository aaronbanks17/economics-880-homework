using Parameters, Plots #import the libraries we want
include("deterministic_growth_functions.jl") #import the functions that solve our growth model
prim, res = Initialize() #initialize primitive and results structs
@time Solve_model(prim, res) #solve the model!

@unpack val_func, pol_func = res
@unpack k_grid, z_grid = prim

##############Make plots
mkpath("output")

#value function
plot(k_grid, val_func, title="Value Function V(K,Z)",ylabel = "value V(K,Z)", label = permutedims(["Z = $(z)" for z in z_grid]),xlabel = "capital K")
savefig("output/Value_Functions.png")

#policy functions
plot(k_grid, pol_func, title="Policy Function K'(K,Z)",ylabel = "policy K'(K,Z)", label = permutedims(["Z = $(z)" for z in z_grid]),xlabel = "capital K",linestyle=:solid)
plot!(k_grid,k_grid,label = "45 degree",color="red",linestyle=:dash)
savefig("output/Policy_Functions.png")

#changes in policy function
pol_func_δ = pol_func.-k_grid
plot(k_grid, pol_func_δ, title="Saving Policy Function K'(K,Z) - K",ylabel = "saving policy K'(K,Z) - K", label = permutedims(["Z = $(z)" for z in z_grid]),xlabel = "capital K")
hline!([0], linestyle=:dash, color=:black, label="")
savefig("output/Policy_Functions_Changes.png")

println("All done!")
################################