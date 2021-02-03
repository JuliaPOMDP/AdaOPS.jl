@testset "Independent Bounds" begin
    m = BabyPOMDP()
    planner = solve(AdaOPSSolver(), m)
    belief = WPFBelief([true], [1.0], 1, 1, AdaOPSTree(planner, initialstate(m)), true)
    b = IndependentBounds(0.0, -1e-5)
    @test bounds(b, m, belief, 90, true) == (0.0, -1e-5)
    b = IndependentBounds(0.0, -1e-5, consistency_fix_thresh=1e-5)
    @test bounds(b, m, belief, 90, true) == (0.0, 0.0)
    b = IndependentBounds(0.0, -1e-4, consistency_fix_thresh=1e-5)
    @test bounds(b, m, belief, 90, true) == (0.0, -1e-4)
end
