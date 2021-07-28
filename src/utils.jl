using Static

as_bool(b::Bool) = b
as_bool(b::StaticBool) = Bool(b)

