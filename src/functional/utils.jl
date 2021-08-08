function split_head(head, x)
    hs, rem = divrem(size(x, 1), head)
    @assert iszero(rem)
    return reshape(x, hs, head, Base.tail(size(x))...)
end

move_head_dim_out_perm(x, nobatch=static(false)) = move_head_dim_out_perm(static(ndims(x)), nobatch)
@inline function move_head_dim_out_perm(n::Integer, nobatch::Union{Bool, StaticBool}=static(false))
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

function move_head_dim_out(x, nobatch=static(false))
    perm = move_head_dim_out_perm(x, nobatch)
    return permutedims(x, perm)
end

move_head_dim_in_perm(x, nobatch=static(false)) = move_head_dim_in_perm(static(ndims(x)), nobatch)
@inline function move_head_dim_in_perm(n::Integer, nobatch::Union{Bool, StaticBool}=static(false))
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

function move_head_dim_in(x, nobatch=static(false))
    perm = move_head_dim_in_perm(x, nobatch)
    return permutedims(x, perm)
end

function merge_head(x)
    return reshape(x, Base.setindex(Base.tail(size(x)), :, 1))
end
