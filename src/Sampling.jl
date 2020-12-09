function KLDSampleSize(k::Int, ζ::Float64 = 0.1)
"""
Return the minimum sample size in order to achieve an error at most ζ with a 95% level of confidence according to KLD-Sampling.
"""
    if k <= 1
        k = 1.2
    end
    a = (k-1)/2
    b = 1/(a*9)
    return (1-b+sqrt(b)*quantile(Normal(), 0.95))^3*a/ζ
end
