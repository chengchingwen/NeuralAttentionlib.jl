const Container{T} = Union{NTuple{N, T}, Vector{T}} where N

function getmask(ls::Container{<:Container})
    lens = map(length, ls)
    m = zeros(Float32, maximum(lens), length(lens))

    for (i, l) ∈ enumerate(ls)
        selectdim(selectdim(m, 2, i), 1, 1:length(l)) .= 1
    end
    reshape(m, (1, size(m)...))
end

getmask(m1::A, m2::A) where A <: Abstract3DTensor = permutedims(m1, [2,1,3]) .* m2
