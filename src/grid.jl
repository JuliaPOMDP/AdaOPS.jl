import Base.length
mutable struct StateGrid{D}
    convert::Function
    cutPoints::Vector{Vector{Float64}}

    function StateGrid{D}(convert, cutPoints...) where D
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
        return new(convert, newCutPoints)
    end
end
StateGrid(convert, cutPoints...) = StateGrid{Base.length(cutPoints)}(convert, cutPoints...)
Base.length(grid::StateGrid{D}) where D = D

zeros_like(grid::StateGrid{D}) where D = zeros(Int, NTuple{D, Int}(length(points)+1 for points in grid.cutPoints))

function access(grid::StateGrid{D}, access_cnt::Array{Int,D}, s::S, pomdp::POMDP{S}) where {S,D}
    s = grid.convert(s, pomdp)::AbstractVector{Float64}
    ind = zeros(Int, D)
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
        ind[d] = mid_ind
    end
    access_cnt[ind...] += 1
    return access_cnt[ind...] == 1 ? true : false
end