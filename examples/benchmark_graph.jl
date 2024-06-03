using ElectronLiquid, FeynmanDiagram
using BenchmarkTools
using MCIntegration, CSV, DataFrames

dim = 3
rs = 2.0
order = 3
mass2 = 0.5
beta = 10.0
isLayered2D = false
root_dir = "./source_codeParquetAD/"

para = UEG.ParaMC(rs=rs, beta=beta, Fs=-0.0, order=order, mass2=mass2, dim=dim)

partition = [(order, 0, 0)]
name = "sigma"
maxMomNum = order + 1
β, kF = para.dim, para.β, para.kF
kgrid = [kF]
ngrid = [0]
alpha = 3.0

df = CSV.read(root_dir * "loopBasis_$(name)_maxOrder6.csv", DataFrame)
loopBasis = [df[!, col][1:maxMomNum] for col in names(df)]
momLoopPool = FrontEnds.LoopPool(:K, dim, loopBasis)

# Helper function to convert string to integer vector
function _StringtoIntVector(str::AbstractString)
    pattern = r"[-+]?\d+"
    return [parse(Int, m.match) for m in eachmatch(pattern, str)]
end


extT_labels = [[[1, 1], [1, 2], [1, 3]]]
leafstates = Vector{Vector{Propagator.LeafStateAD}}()
leafvalues = Vector{Vector{Float64}}()
for key in partition
    key_str = join(string.(key))
    df = CSV.read(root_dir * "leafinfo_$(name)_$key_str.csv", DataFrame)
    leafstates_par = Vector{Propagator.LeafStateAD}()
    for row in eachrow(df)
        push!(leafstates_par, Propagator.LeafStateAD(row[2], _StringtoIntVector(row[3]), row[4:end]...))
    end
    push!(leafstates, leafstates_par)
    push!(leafvalues, df[!, names(df)[1]])
end

root = zeros(Float64, maximum(length.(extT_labels)))
# root = zeros(Float64, 3)
K = MCIntegration.FermiK(dim, kF, 0.5 * kF, 10.0 * kF, offset=1)
K.data[:, 1] .= 0.0
K.data[1, 1] = kF
# T = MCIntegration.Continuous(0.0, β; grid=collect(LinRange(0.0, β, 1000)), offset=1, alpha=alpha)
T = Continuous(0.0, β; alpha=alpha, adapt=true, offset=1)
T.data[1] = 0.0
X = MCIntegration.Discrete(1, 1, alpha=alpha)
ExtKidx = MCIntegration.Discrete(1, 1, alpha=alpha)

dof = [[order, order - 1, 1, 1]] # K, T, X, ExtKidx
# observable of sigma diagram of different permutations
obs = [zeros(ComplexF64, 1, 1) for _ in 1:length(dof)]
vars = (K, T, X, ExtKidx)

config = Configuration(;
    var=vars,
    dof=dof,
    type=ComplexF64, # type of the integrand
    obs=obs,
    userdata=(para, kgrid, ngrid, maxMomNum, extT_labels, leafstates, leafvalues, momLoopPool, root, isLayered2D, partition)
)


function rand_vars(vars)
    vars[1].data[:, 2:end] .= randn(size(vars[1].data[:, 2:end]))
    vars[2].data[2:end] .= rand(size(vars[2].data[2:end])[1])
end

Sigma.integrandKW_Clib(1, vars, config)

@btime Sigma.integrandKW_Clib(1, vars, config)

@benchmark Sigma.integrandKW_Clib(1, vars, config) setup = (rand_vars(vars))
println("Finished benchmarking Sigma.integrandKW_Clib")