using ElectronLiquid
using CompositeGrids

dim = 3
rs = [2.0]
order = [3]  # maximum diagram order of the run
mass2 = [0.5]

Fs = [-0.0]        # Fermi liquid parameter with zero angular momentum
beta = [50.0]      # inverse temperature beta = β*E_F 
neval = 4e7       # number of Monte Carlo samples
isDynamic = false  # whether to use effective field theory with dynamic screening or not 
isFock = false     # whether to use Fock renormalization or not

diagGenerate = :GV   # :GV or :Parquet, algorithm to generate diagrams
isLayered2D = false  # whether to use layered 2D system or not
spin = 2    # 2 for unpolarized, 1 for polarized
ispolarized = spin < 2

is_Clib = true
# is_Clib = false

for (_rs, _mass2, _F, _beta, _order) in Iterators.product(rs, mass2, Fs, beta, order)
    para = ParaMC(rs=_rs, beta=_beta, Fs=_F, order=_order, mass2=_mass2, isDynamic=isDynamic, dim=dim, isFock=isFock, spin=spin)
    println(UEG.short(para))
    kF = para.kF
    kgrid = [kF,]
    # ngrid = [-1, 0, 1]
    ngrid = [0]

    # partition = UEG.partition(_order)
    # reweight_goal = Float64[]
    # for (order, sOrder, vOrder) in partition
    #     reweight_factor = 2.0^(2order + 2sOrder + vOrder - 2)
    #     if (order, sOrder, vOrder) == (1, 0, 0)
    #         reweight_factor = 4.0
    #     end
    #     push!(reweight_goal, reweight_factor)
    # end
    # push!(reweight_goal, 4.0)
    partition = [(3, 0, 0)]
    neighbor = [(1, 2)]

    # filename = mission == "Z" ? sigma_z_filename : sigma_k_filename
    filename = "dataZ_test.jld2"
    if is_Clib
        sigma, result = Sigma.MC_Clib(para; kgrid=kgrid, ngrid=ngrid,
            neval=neval, filename=filename, partition=partition, neighbor=neighbor,
            # reweight_goal=reweight_goal,
            isLayered2D=isLayered2D)
    else
        sigma, result = Sigma.MC(para; kgrid=kgrid, ngrid=ngrid, diag_generator=diagGenerate,
            neval=neval, filename=filename, partition=partition, neighbor=neighbor,
            #reweight_goal=reweight_goal,
            isLayered2D=isLayered2D)
    end
end
