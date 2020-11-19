struct TextTree
    children::Vector{Vector{Int}}
    text::Vector{String}
end

function TextTree(D::OPSTree)
    lenb = length(D.children)
    lenba = length(D.ba_children)
    len = lenb + lenba
    m = n_particles(D.root_belief)
    children = Vector{Vector{Int}}(len)
    text = Vector{String}(len)
    for b in 1:lenb
        children[b] = D.children[b] .+ lenb
        text[b] = @sprintf("o:%-5s U:%6.2f, L:%6.2f, ESS:%6.2f",
                           b==1 ? "<root>" : string(D.obs[b]),
                           D.U[b],
                           D.L[b],
                           sum(D.weights[b])^2/(dot(D.weights[b], D.weights[b])),
                            )
    end
    for ba in 1:lenba
        children[ba+lenb] = D.ba_children[ba]
        text[ba+lenb] = @sprintf("a:%-5s L:%6.2f U:%6.2f, r:%6.2f", D.ba_action[ba], D.ba_L[ba], D.ba_U[ba], D.ba_Rsum[ba]/m)
    end
    return TextTree(children, text)
end

struct TreeView
    t::TextTree
    root::Int
    depth::Int
end

TreeView(D::OPSTree, b::Int, depth::Int) = TreeView(TextTree(D), b, depth)

Base.show(io::IO, tv::TreeView) = shownode(io, tv.t, tv.root, tv.depth, "", "")

function shownode(io::IO, t::TextTree, n::Int, depth::Int, item_prefix::String, prefix::String)
    print(io, item_prefix)
    print(io, @sprintf("[%-4d]", n))
    print(io, " $(t.text[n])")
    if depth <= 0
        println(io, " ($(length(t.children[n])) children)")
    else
        println(io)
        if !isempty(t.children[n])
            for c in t.children[n][1:end-1]
                shownode(io, t, c, depth-1, prefix*"├──", prefix*"│  ")
            end
            shownode(io, t, t.children[n][end], depth-1, prefix*"└──", prefix*"   ")
        end
    end
end
