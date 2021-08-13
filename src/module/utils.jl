"import and export"
macro imexport(body)
    importf = body.head
    @assert importf == :using || importf == :import
    args = body.args[]
    @assert args.head == :(:)
    ex = Expr(:export, map(x->x.args[], args.args[2:end])...)
    return quote
        $body
        $ex
    end
end
