module NeuralAttentionlibFiniteDifferences

using NeuralAttentionlib
using NeuralAttentionlib: CollapsedDimsArray, collapseddims
using FiniteDifferences

function FiniteDifferences.to_vec(X::CollapsedDimsArray)
    x_vec, back = to_vec(collapseddims(X))
    s = size(parent(X))
    ni = X.ni
    nj = X.nj
    function CollapsedDimsArray_from_vec(x_vec)
        return CollapsedDimsArray(reshape(back(x_vec), s), ni, nj)
    end
    return x_vec, CollapsedDimsArray_from_vec
end

end
