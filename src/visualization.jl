function D3Trees.D3Tree(D::OPSTree; title="OPS Tree", kwargs...)
    lenb = length(D.children)
    lenba = length(D.ba_children)
    len = lenb + lenba
    m = n_particles(D.root_belief)
    children = Vector{Vector{Int}}(undef, len)
    text = Vector{String}(undef, len)
    tt = fill("", len)
    link_style = fill("", len)
    for b in 1:lenb
        children[b] = D.children[b] .+ lenb
        text[b] = @sprintf("""
                           o:%s Effectiveness Indicator:%6.2f
                           L:%6.2f, U:%6.2f""",
                           b==1 ? "<root>" : string(D.obs[b]),
                           1/(m*dot(D.weights[b], D.weights[b])),
                           D.L[b],
                           D.U[b],
                          )
        tt[b] = """
                o: $(b==1 ? "<root>" : string(D.obs[b]))
                Effectiveness Indicator: $(1/(m*dot(D.weights[b], D.weights[b])))
                L: $(D.L[b])
                U: $(D.U[b])
                $(length(D.children[b])) children
                """
        link_width = 20.0
        link_style[b] = "stroke-width:$link_width"
        for ba in D.children[b]
            link_style[ba+lenb] = "stroke-width:$link_width"
        end

        for ba in D.children[b]
            children[ba+lenb] = D.ba_children[ba]
            text[ba+lenb] = @sprintf("""
                                     a:%s r:%6.2f
                                     L:%6.2f, U:%6.2f""",
                                     D.ba_action[ba], D.ba_Rsum[ba]/m,
                                     D.ba_L[ba], D.ba_U[ba],
                                     )
            tt[ba+lenb] = """
                          a: $(D.ba_action[ba])
                          r: $(D.ba_Rsum[ba]/m)
                          L: $(D.ba_L[ba])
                          U: $(D.ba_U[ba])
                          $(length(D.ba_children[ba])) children
                          """
        end

    end
    return D3Tree(children;
                  text=text,
                  tooltip=tt,
                  link_style=link_style,
                  title=title,
                  kwargs...
                 )
end

Base.show(io::IO, mime::MIME"text/html", D::OPSTree) = show(io, mime, D3Tree(D))
Base.show(io::IO, mime::MIME"text/plain", D::OPSTree) = show(io, mime, D3Tree(D))

"""
Fill all the elements of the cache for b and children of b and return L[b]
"""