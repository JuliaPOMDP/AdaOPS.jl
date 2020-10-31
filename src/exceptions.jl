struct NoGap <: NoDecision
    value::Float64
end

Base.show(io::IO, ng::NoGap) = print(io, """
    The lower and upper bounds for the root belief were both $(ng.value), so no tree was created.

    Use the default_action solver parameter to specify behavior for this case.
    """)
