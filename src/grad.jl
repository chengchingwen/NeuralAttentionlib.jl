using FiniteDifferences

function FiniteDifferences.to_vec(X::CollapsedDimArray)
    x_vec, back = to_vec(collapseddim(X))
    s = size(parent(X))
    si = X.si
    sj = X.sj
    function CollapsedDimArray_from_vec(x_vec)
        return CollapsedDimArray(reshape(back(x_vec), s), si, sj)
    end
    return x_vec, CollapsedDimArray_from_vec
end
