"""
KLDSampleSize(k::Int, ζ, η::Float64 = 0.05)

Return the minimum sample size in order to achieve an error at most ζ with a 1-η level of confidence according to KLD-Sampling.
"""
function KLDSampleSize(k::Int, ζ, η::Float64 = 0.05)
    k = convert(Float64, k)
    if k <= 1.0
        k = 1.5
    end
    a = (k-1.0)/2.0
    b = 1.0/(a*9.0)
    return (1.0-b+sqrt(b)*quantile(Normal(), 1-η))^3.0*a/ζ
end