function D3Trees.D3Tree(D::AdaOPSTree; title="AdaOPS Tree", kwargs...)
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
                           o:%s ESS:%6.2f
                           l:%6.2f, u:%6.2f""",
                           b==1 ? "<root>" : string(D.obs[b]),
                           sum(D.weights[b])^2/dot(D.weights[b], D.weights[b]),
                           D.l[b],
                           D.u[b],
                          )
        tt[b] = """
                o: $(b==1 ? "<root>" : string(D.obs[b]))
                ESS: $(sum(D.weights[b])^2/dot(D.weights[b], D.weights[b]))
                l: $(D.l[b])
                u: $(D.u[b])
                $(length(D.children[b])) children
                """
        link_width = 2.0
        link_style[b] = "stroke-width:$link_width"
        for ba in D.children[b]
            link_style[ba+lenb] = "stroke-width:$link_width"
        end

        for ba in D.children[b]
            children[ba+lenb] = D.ba_children[ba]
            text[ba+lenb] = @sprintf("""
                                     a:%s r:%6.2f
                                     l:%6.2f, u:%6.2f""",
                                     D.ba_action[ba], D.ba_r[ba],
                                     D.ba_l[ba], D.ba_u[ba],
                                     )
            tt[ba+lenb] = """
                          a: $(D.ba_action[ba])
                          r: $(D.ba_r[ba])
                          l: $(D.ba_l[ba])
                          u: $(D.ba_u[ba])
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

Base.show(io::IO, mime::MIME"text/html", D::AdaOPSTree) = show(io, mime, D3Tree(D))
Base.show(io::IO, mime::MIME"text/plain", D::AdaOPSTree) = show(io, mime, D3Tree(D))