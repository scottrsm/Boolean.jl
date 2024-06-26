using Boolean
using Test


@testset "Boolean (Fidelity)              " begin
    @test length(detect_ambiguities(Boolean)) == 0
end

@testset "Boolean (Formulas)              " begin

	# Package data directory.
	REPO_DATA_DIR = joinpath(@__DIR__, "../data")

    # Set up
    init_logic(3)

    # Tests Functions
    @test create_bool_rep("z1 + z2") == Blogic("z1 + z2", "z", BitVector(Bool[0, 1, 1, 1, 0, 1, 1, 1]))
    @test BitArray([((i - 1) >> (j - 1)) & 1 for i in 1:2^3, j in 1:3]) == BitMatrix(Bool[0 0 0;
        1 0 0;
        0 1 0;
        1 1 0;
        0 0 1;
        1 0 1;
        0 1 1;
        1 1 1])

    # Try parsing logic.
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
    @test simplifyLogic(parseLogic("x1 ⊕ (x2 + x3) ⊕ x4 ⊕ x5")) == :(⊕(x1, x4, x5, x2 + x3))

    # Test Outer Constructor.
    f = Blogic("(x1 + x2) * x3")
    @test f.val == Bool[0, 0, 0, 0, 0, 1, 1, 1]

    # Test Blogic creation from file.
	f = Blogic_from_file(joinpath(REPO_DATA_DIR, "example1.txt"))
    @test f.val == Bool[0, 1, 1, 1, 1, 1, 1, 1]

end


@testset "Boolean (Macros and Equivalence)" begin
    f = @bfunc x1 + x2 ⟹ x3
    @test f.val == Bool[1, 0, 0, 0, 1, 1, 1, 1]

    f = @bfunc begin
        x1 + x2
        ⟹
        x2 * x3
    end
    @test f.val == Bool[1, 0, 0, 0, 1, 0, 1, 1]


    f1 = @bfunc x1 + x2 * x3
    f2 = @bfunc x1 * x3 ⊕ x2 * x3

    @test isEquiv(f1, f1)
    @test ~isEquiv(f1, f2)

    f1 = @bfunc (x1 + x2) ⟺ x2 * x3
    f2 = @bfunc ~(x1 + x2) ⊕ x2 * x3

    @test isEquiv(f1, f2)

end

