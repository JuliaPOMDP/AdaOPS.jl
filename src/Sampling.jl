function KLDSampleSize(k::Int, ζ::Float64 = 0.1, η::Float64 = 0.95)
"""
Return the minimum sample size in order to achieve an error at most ζ with a 95% level of confidence according to KLD-Sampling.
"""
    k = convert(Float64, k)
    if k <= 1.0
        k = 1.2
    end
    a = (k-1.0)/2.0
    b = 1.0/(a*9.0)
    return (1.0-b+sqrt(b)*quantile(Normal(), η))^3.0*a/ζ
end
