module LongRangeHamiltonians

export ExponentialExpansionAlgorithm, HankelExpansion, LsqExpansion, exponential_expansion, expansion_error
export ExponentialDecayTerm, GenericDecayTerm, PowerlawDecayTerm, SchurMPOTensor


using LsqFit
using LinearAlgebra: qr, pinv, eigvals
using SphericalTensors
const TK = SphericalTensors
using DMRG, InfiniteDMRG

# long range Hamiltonians
include("longrange.jl")
include("exponentialdecay.jl")
include("exponentialexpansion.jl")
include("generaldecay.jl")

end