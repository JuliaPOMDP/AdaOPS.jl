mutable struct StateGrid{D}
    cutPoints::Vector{Vector{Float64}}

    function StateGrid{D}(cutPoints...) where D
        newCutPoints = Array{Vector{Float64}}(undef, length(cutPoints))
        for i = 1:D
            if length(Set(cutPoints[i])) != length(cutPoints[i])
                error(@sprintf("Duplicates cutpoints are not allowed (duplicates observed in dimension %d)",i))
            end
            if !issorted(cutPoints[i])
                error("Cut points must be sorted")
            end
            newCutPoints[i] = cutPoints[i]
        end
        return new(newCutPoints)
    end
end

StateGrid(cutPoints...) = StateGrid{length(cutPoints)}(cutPoints...)

zeros_like(grid::StateGrid) = zeros(Int64, [length(points)+1 for points in grid.cutPoints]...)
zeros_like(::Nothing) = nothing

function access(grid::StateGrid, access_cnt::Array, s, pomdp::POMDP)
    s = convert_s(AbstractVector{Float64}, s, pomdp)
    ind = zeros(Int64, length(grid.cutPoints))
    for d in 1:length(grid.cutPoints)
        cutPoints = grid.cutPoints[d]
        # Binary search for the apt grid
        start_ind = 1
        end_ind = length(cutPoints) + 1
        mid_ind = floor(Int64, (start_ind+end_ind)/2)
        while start_ind < end_ind
            cutPoint = cutPoints[mid_ind]
            if s[d] < cutPoint
                end_ind = mid_ind
            else
                start_ind = mid_ind + 1
            end
            mid_ind = floor(Int64, (start_ind+end_ind)/2)
        end
        ind[d] = mid_ind
    end
    access_cnt[ind...] += 1
    return access_cnt[ind...] == 1 ? true : false
end
access(grid::Nothing, access_cnt, s, pomdp) = false