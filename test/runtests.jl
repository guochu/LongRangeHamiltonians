push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/InfiniteDMRG/src")
push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/GeneralHamiltonians/src")

push!(LOAD_PATH, dirname(Base.@__DIR__) * "/src")

using Test, Random
using DMRG, InfiniteDMRG, GeneralHamiltonians
using LongRangeHamiltonians

Random.seed!(1342)

include("util.jl")

## algorithms
include("longrange.jl")
