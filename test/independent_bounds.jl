@testset "Independent Bounds" begin
    m = BabyPOMDP()
    belief = WPFBelief([true], [1.0], true)
    b = IndependentBounds(0.0, -1e-5)
    @test bounds(b, m, belief) == (0.0, -1e-5)
    b = IndependentBounds(0.0, -1e-5, consistency_fix_thresh=1e-5)
    @test bounds(b, m, belief) == (0.0, 0.0)
    b = IndependentBounds(0.0, -1e-4, consistency_fix_thresh=1e-5)
    @test bounds(b, m, belief) == (0.0, -1e-4)
end
