as_collapsed(x::AbstractVector) = CollapsedDimsArray(x, static(0), static(0))
as_collapsed(x::AbstractMatrix) = CollapsedDimsArray(x, static(1), static(0))
as_collapsed(x::AbstractArray) = CollapsedDimsArray(x, static(ndims(x)) - static(2), static(1))
as_collapsed(x::CollapsedDimsArray) = x

function split_head(head::Integer, x)
    hs, rem = divrem(size(x, 1), head)
    @assert iszero(rem)
    return reshape(x, hs, head, Base.tail(size(x))...)
end

function split_head(head::Integer, x::CollapsedDimsArray)
    s1 = collapsed_size(x, 1)
    hs, rem = divrem(s1, head)
    @assert iszero(rem)
    _, len_d, batch_d = noncollapsed_size(x)
    y = reshape(parent(x), hs, head, len_d..., batch_d...)
    return CollapsedDimsArray(y, (s1, prod(len_d), prod(batch_d)), static(length(len_d)), static(length(batch_d)))
end

function move_head_dim_out_perm(x::CollapsedDimsArray)
    return (1, ntuple(Base.Fix1(+, 2), x.ni)..., 2, ntuple(Base.Fix1(+, 2 + x.ni), x.nj)...)
end

move_head_dim_out_perm(x::AbstractArray, nobatch = static(false)) = move_head_dim_out_perm(ndims(x), nobatch)
move_head_dim_out_perm(n::Integer, nobatch = static(false)) = move_head_dim_out_perm(static(n), nobatch)
@inline function move_head_dim_out_perm(n::StaticInt, nobatch::Union{Bool, StaticBool} = static(false))
    if as_bool(nobatch)
        perm = let N = Int(n - 1)
            (1, ntuple(i->mod(i,N)+2, N)...)
        end
    else
        perm = let N = Int(n - 2)
            (1, ntuple(i->mod(i,N)+2, N)..., N+2)
        end
    end  
    return perm
end

function move_head_dim_out(x::CollapsedDimsArray)
    perm = move_head_dim_out_perm(x)
    return CollapsedDimsArray(permutedims(parent(x), perm), x.ni, x.nj + static(1))
end

function move_head_dim_out(x, nobatch=static(false))
    perm = move_head_dim_out_perm(x, nobatch)
    return permutedims(x, perm)
end

move_head_dim_in_perm(x::AbstractArray, nobatch = static(false)) = move_head_dim_in_perm(ndims(x), nobatch)
move_head_dim_in_perm(n::Integer, nobatch = static(false)) = move_head_dim_in_perm(static(n), nobatch)
@inline function move_head_dim_in_perm(n::StaticInt, nobatch::Union{Bool, StaticBool} = static(false))
    if as_bool(nobatch)
        perm = let N = n - 1
            (1, ntuple(i->mod(i-2,N)+2, N)...)
        end
    else
        perm = let N = n - 2
            (1, ntuple(i->mod(i-2,N)+2, N)..., N+2)
        end
    end  
    return perm
end

function move_head_dim_in(x, nobatch = static(false))
    perm = move_head_dim_in_perm(x, nobatch)
    return permutedims(x, perm)
end

function merge_head(x)
    return reshape(x, Base.setindex(Base.tail(size(x)), :, 1))
end
