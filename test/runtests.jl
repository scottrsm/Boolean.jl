using Test
using Boolean


@testset "Boolean (Fidelity)" begin
    @test length(detect_ambiguities(Boolean)) == 0
end

@testset "Boolean (Formulas)" begin
    ## Set up
	init_logic(Int64(3))
    f1 = "x1 + x2 * x3"
    f2 = "(x1 * x3) ⊕ (x2 * x3)"


    ## Tests Functions
    @test create_bool_rep("z1 + z2") == Blogic("z1 + z2", "z", BitVector(Bool[0, 1, 1, 1, 0, 1, 1, 1]))
    @test BitArray([((i-1) >> (j-1)) & 1  for i in 1:2^3, j in 1:3]) == BitMatrix(Bool[ 0  0  0;
                                                                                        1  0  0;
                                                                                        0  1  0;
                                                                                        1  1  0;
                                                                                        0  0  1;
                                                                                        1  0  1;
                                                                                        0  1  1;
                                                                                        1  1  1])

	@test simplifyLogic(parseLogic("x1 * (1 ⊕ x1)")) == :(x1 * ~x1)
	@test simplifyLogic(parseLogic("(z1 ⊕ z3) ⊕ (0 ⊕ z3)")) == :z1
	@test simplifyLogic(parseLogic("((x1 + x2) * x3) * (x1 * 0 + x2 * 1)")) == :(x2 * (x3 * (x1 + x2)))
	@test simplifyLogic(parseLogic("0 + x1")) == :x1
	@test simplifyLogic(parseLogic("x1 * (x1 + 0)")) == :x1
	@test simplifyLogic(parseLogic("x1 * (x1 + 1)")) == :x1
	@test simplifyLogic(parseLogic("((0 + x2) * x3) * x2")) == :(x2 * (x2 * x3))
	@test simplifyLogic(parseLogic("((1 + x2) * x3) * x2")) == :(x2 * x3)
	@test simplifyLogic(parseLogic("(x1 * x2) + (x2 * x3) + (x1 * x2)")) == :(x1 * x2 + x2 * x3)
	@test simplifyLogic(parseLogic("(x1 * x2) + (x2 * x3) + (x1 * x2)")) == :(x1 * x2 + x2 * x3) 
	@test simplifyLogic(parseLogic("((1 + x2) * x3) * x2")) == :(x2 * x3)
	@test simplifyLogic(parseLogic("x1 ⊕ (x2 + x3) ⊕ x4 ⊕ x5")) ==  :(⊕(x1, x4, x5, x2 + x3))

    @test isEquiv(f1, f1)
    @test ~ isEquiv(f1,f2)
end

