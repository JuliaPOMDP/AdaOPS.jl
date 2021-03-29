import Base.length
mutable struct StateGrid{D}
    cutPoints::Vector{Vector{Float64}}
    indicies::Vector{Int}

    function StateGrid{D}(cutPoints...) where D
        newCutPoints = Vector{Vector{Float64}}(undef, length(cutPoints))
        for i = 1:D
            if length(Set(cutPoints[i])) != length(cutPoints[i])
                error(@sprintf("Duplicates cutpoints are not allowed (duplicates observed in dimension %d)",i))
            end
            if !issorted(cutPoints[i])
                error("Cut points must be sorted")
            end
            newCutPoints[i] = cutPoints[i]
        end
        indicies = cumprod([1; [length(points)+1 for points in newCutPoints[1:end-1]]...])
        return new(newCutPoints, indicies)
    end
end
StateGrid(cutPoints...) = StateGrid{Base.length(cutPoints)}(cutPoints...)
Base.length(grid::StateGrid{D}) where D = D
Base.size(grid::StateGrid{D}) where D = NTuple{D, Int}(length(points)+1 for points in grid.cutPoints)

zeros_like(grid::StateGrid{D}) where D = zeros(Int, NTuple{D, Int}(length(points)+1 for points in grid.cutPoints))

function access(grid::StateGrid{D}, access_cnt::Array{Int,D}, s::S, pomdp::POMDP{S}) where {S,D}
    s = convert(SVector{D,Float64}, s)
    index = 1
    for d in 1:D
        cutPoints = grid.cutPoints[d]
        # Binary search for the apt grid
        start_ind = 1
        end_ind = length(cutPoints) + 1
        mid_ind = div(start_ind+end_ind, 2)
        while start_ind < end_ind
            cutPoint = cutPoints[mid_ind]
            if s[d] < cutPoint
                end_ind = mid_ind
            else
                start_ind = mid_ind + 1
            end
            mid_ind = div(start_ind+end_ind, 2)
        end
        index += (mid_ind-1) * grid.indicies[d]
    end
    access_cnt[index] += 1
    return access_cnt[index] == 1 ? true : false
end