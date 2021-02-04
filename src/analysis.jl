function info_analysis(info::Dict)
    D = get(info, :tree, nothing)
    if D !== nothing
        show(stdout, MIME("text/plain"), D)
        println("Number of action node expanded:", D.ba)
        println("Number of belief node expanded:", D.b)
        m = length.(view(D.ba_particles, 1:D.ba))
        println(@sprintf("m: mean±std = %5.2f±%4.2f", mean(m), std(m)))
        println("Confidence interval (0.1, 0.9) = ", quantile(m, (0.1, 0.9)))
        branch = length.(view(D.ba_children, 1:D.ba))
        println(@sprintf("Number of observation branchs: mean±std = %5.2f±%4.2f", mean(branch), std(branch)))
        println("Confidence interval (0.1, 0.9) = ", quantile(branch, (0.1, 0.9)))
    end
    depth = info[:depth]
    println("Times of exploration: ", length(depth))
    println(@sprintf("Depth of exploration: mean±std = %5.2f±%4.2f", mean(depth), std(depth)))
    println("Confidence interval (0.1, 0.9) = ", quantile(depth, (0.1, 0.9)))
    return nothing
end

function hist_analysis(hist::H) where H<:AbstractSimHistory
    theme(:wong)
    infos = ainfo_hist(hist)

    mean_d = Float64[]
    std_d = Float64[]
    for info in infos
        depth = info[:depth]
        push!(mean_d, mean(depth))
        push!(std_d, std(depth))
    end
    p1 = plot(mean_d, ribbon=(mean_d.-std_d, mean_d.+std_d), xaxis="Steps", yaxis="Depth of exploration", legend=false)

    D = get(first(infos), :tree, nothing)
    if D === nothing
        display(p1)
    else
        num_anode = Int[]
        num_bnode = Int[]
        mean_m = Float64[]
        std_m = Float64[]
        mean_branch = Float64[]
        std_branch = Float64[]

        for info in infos
            D = info[:tree]
            push!(num_anode, D.ba)
            push!(num_bnode, D.b)
            m = length.(view(D.ba_particles, 1:D.ba))
            push!(mean_m, mean(m))
            push!(std_m, std(m))
            branch = length.(view(D.ba_children, 1:D.ba))
            push!(mean_branch, mean(branch))
            push!(std_branch, std(branch))
        end
        p2 = plot(hcat(num_anode,num_bnode), label=["Action" "Belief"], xaxis="Steps", yaxis="Nodes expanded")
        p3 = plot(mean_m, ribbon=(mean_m.-std_m, mean_m.+std_m), xaxis="Steps", yaxis="Avg. particles", legend=false)
        p4 = plot(mean_branch, ribbon=(mean_branch.-std_branch, mean_branch.+std_branch), xaxis="Steps", yaxis="Avg. Obs.", legend=false)
        display(plot(p1, p2, p3, p4, layout = (2, 2)))
    end
    return nothing
end