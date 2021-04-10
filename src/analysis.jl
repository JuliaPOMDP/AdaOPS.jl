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

function hist_analysis(hist::H, display_mean_and_std::Bool = false) where H<:AbstractSimHistory
    infos = ainfo_hist(hist)

    median_d = Float64[]
    lower_d = Float64[]
    upper_d = Float64[]
    mean_d = Float64[]
    std_d = Float64[]

    for info in infos
        depth = info[:depth]
        l_d, m_d, u_d = quantile(depth, (0.1, 0.5, 0.9))
        push!(median_d, m_d)
        push!(lower_d, m_d-l_d)
        push!(upper_d, u_d-m_d)
        push!(mean_d, mean(depth))
        push!(std_d, std(depth))
    end
    p1 = plot(median_d, ribbon=(lower_d, upper_d), xaxis="Steps", yaxis="Depth of exploration", label="quantile")
    if display_mean_and_std
        plot!(p1, mean_d, ribbon=std_d, label="mean", legend=:best)
    end

    D = get(first(infos), :tree, nothing)
    if D === nothing
        display(p1)
    else
        num_anode = Int[]
        num_bnode = Int[]

        median_m = Float64[]
        lower_m = Float64[] # lower quantile
        upper_m = Float64[] # upper quantile
        mean_m = Float64[]
        std_m = Float64[]

        median_branch = Float64[]
        lower_branch = Float64[]
        upper_branch = Float64[]
        mean_branch = Float64[]
        std_branch = Float64[]

        for info in infos
            D = info[:tree]
            push!(num_anode, D.ba)
            push!(num_bnode, D.b)
            m = length.(view(D.ba_particles, 1:D.ba))
            l_m, m_m, u_m = quantile(m, (0.1, 0.5, 0.9))
            push!(median_m, m_m)
            push!(lower_m, m_m-l_m)
            push!(upper_m, u_m-m_m)
            push!(mean_m, mean(m))
            push!(std_m, std(m))
            branch = length.(view(D.ba_children, 1:D.ba))
            l_b, m_b, u_b = quantile(branch, (0.1, 0.5, 0.9))
            push!(median_branch, m_b)
            push!(lower_branch, m_b-l_b)
            push!(upper_branch, u_b-m_b)
            push!(mean_branch, mean(branch))
            push!(std_branch, std(branch))
        end
        p2 = plot(hcat(num_anode,num_bnode), label=["Action" "Belief"], xaxis="Steps", yaxis="Nodes expanded", legend=:best)
        p3 = plot(median_m, ribbon=(lower_m, upper_m), xaxis="Steps", yaxis="Particles used", label="quantile")
        p4 = plot(median_branch, ribbon=(lower_branch, upper_branch), xaxis="Steps", yaxis="Obs. Num.", label="quantile")
        if display_mean_and_std
            plot!(p3, mean_m, ribbon=std_m, label="mean", legend=:best)
            plot!(p4, mean_branch, ribbon=std_branch, label="mean", legend=:best)
        end
        display(plot(p1, p2, p3, p4, layout = (2, 2)))
    end
    return nothing
end