using LinearAlgebra: tril!, triu!

function batched_tril!(x::A, d) where {T, N, A <: AbstractArray{T, N}}
    if N < 2
        error("MethodError: no method matching tril!(::Array{Float64,1})")
    elseif N == 2
        return tril!(x, d)
    else
        s = size(x)
        m = (s[1], s[2])
        ms = s[1] * s[2]
        len = Int(length(x) // ms)
        C = CartesianIndices(Base.tail(Base.tail(s)))
        for i = 1:len
            tril!(@view(x[:, :, C[i]]), d)
        end
        return x
    end
end

function batched_triu!(x::A, d) where {T, N, A <: AbstractArray{T, N}}
    if N < 2
        error("MethodError: no method matching triu!(::Array{Float64,1})")
    elseif N == 2
        return triu!(x, d)
    else
        s = size(x)
        m = (s[1], s[2])
        ms = s[1] * s[2]
        len = Int(length(x) // ms)
        C = CartesianIndices(Base.tail(Base.tail(s)))
        for i = 1:len
            triu!(@view(x[:, :, C[i]]), d)
        end
        return x
    end
end
