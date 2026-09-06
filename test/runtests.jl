using Boolean
using Test
import Random


@testset "Boolean (Fidelity)              " begin
    @test length(detect_ambiguities(Boolean)) == 0
end

@testset "Boolean (Formulas)              " begin

	# Package data directory.
	REPO_DATA_DIR = joinpath(@__DIR__, "../data")

    # Set up: truth tables use at least 3 variables.
    init_logic(3)

    # Tests Functions
    @test create_bool_rep("z1 + z2") == Blogic("z1 + z2", "z", BitVector(Bool[0, 1, 1, 1, 0, 1, 1, 1]))
    @test bool_var_rep(3) == BitMatrix(Bool[0 0 0;
        1 0 0;
        0 1 0;
        1 1 0;
        0 0 1;
        1 0 1;
        0 1 1;
        1 1 1])
    @test BitArray([((i - 1) >> (j - 1)) & 1 for i in 1:2^3, j in 1:3]) == bool_var_rep(3)
    @test_throws DomainError bool_var_rep(0)
    @test_throws DomainError bool_var_rep(23)

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
    @test simplifyLogic(parseLogic("x1 ⊕ x1")) == 0
    @test simplifyLogic(parseLogic("x1 + ~x1 * 0 + 1")) == 1
    @test simplifyLogic(parseLogic("~(~x1)")) == :x1
    @test simplifyLogic(parseLogic("x1 ⟹ x2")) == :(x2 + ~x1)
    @test simplifyLogic(parseLogic("x1 ⟺ x2")) == :(x2 ⊕ ~x1)

    # Test Outer Constructor.
    f = Blogic("(x1 + x2) * x3")
    @test f.val == Bool[0, 0, 0, 0, 0, 1, 1, 1]
    @test f.size == 3 && f.var == "x"

    # Test Blogic creation from file.
	f = Blogic_from_file(joinpath(REPO_DATA_DIR, "example1.txt"))
    @test f.val == Bool[0, 1, 1, 1, 1, 1, 1, 1]
    @test_throws ArgumentError Blogic_from_file(joinpath(REPO_DATA_DIR, "does-not-exist.txt"))

    # Constants in formulas, and formulas that simplify to constants.
    @test create_bool_rep("x1 + 1").val == trues(8)
    @test Blogic("x1 * 0").val == falses(8)
    @test Blogic("x1 ⊕ x1"; simplify=true).val == falses(8)
    @test Blogic("x1 ⊕ x1").val == falses(8)
    @test Blogic("x1 * (1 ⊕ x2)").val == Bool[0, 1, 0, 0, 0, 1, 0, 0]
    @test create_bool_rep("x1 * (1 ⊕ x2)"; simplify=true).val == Bool[0, 1, 0, 0, 0, 1, 0, 0]

    # Multi-line formulas and the `nvars` keyword.
    @test create_bool_rep("x1 + x2\n *\nx2 + x3").val == f.val
    @test create_bool_rep("z1 + z2"; nvars=5).size == 5

    # Reset the minimum size: the size is then the highest variable index.
    init_logic(0)
    @test Blogic("x1 + x2").size == 2 && Blogic("x1 + x2").val == Bool[0, 1, 1, 1]
    @test Blogic("~x1").val == Bool[1, 0]
    @test Blogic("x1 + x5").size == 5
    @test_throws DomainError init_logic(-1)
    @test_throws DomainError init_logic(23)
end


@testset "Boolean (Macros and Equivalence)" begin
    init_logic(0)
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

    # Blogic values are independent of any global state: defining another function,
    # with a different number of variables, does not affect earlier ones.
    f3 = @bfunc (x1 + x2) * x3
    g2 = @bfunc x1 * x2
    @test f3.size == 3 && g2.size == 2
    @test isEquiv(f3, f3) && !isEquiv(f3, g2)
    f4 = @bfunc x1 + x2 + x3 + x4
    @test isEquiv(f4, f4)
    @test isEquiv(@bfunc(x1 + x2), @bfunc(x2 + x1 + 0 * x3))   # different sizes
    @test isEquiv("x1 + x2", "x2 + x1") && isEquiv("x1", "y1") && !isEquiv("x1 * x2", "x1 + x2")
    @test isEquiv("x1 ⟹ x2", "~x1 + x2")

    # Upper case variable names, constants and single variables in the macro.
    @test (@bfunc X1 + X2).val == Bool[0, 1, 1, 1]
    @test (@bfunc x1 ⊕ 1).val == Bool[1, 0]
    @test (@bfunc ~x1).val == Bool[1, 0]
end


@testset "Boolean (Callable Blogic)       " begin
    init_logic(0)
    f3 = @bfunc (x1 + x2) * x3
    g2 = @bfunc x1 * x2

    # Every row of the truth table agrees with the callable form.
    for i in 0:7
        bits = [(i >> k) & 1 for k in 0:2]
        @test f3(bits...) == f3.val[i + 1]
    end
    @test f3(1, 0, 1) && !f3(0, 0, 1) && g2(1, 1) && !g2(1, 0)
    @test f3(true, false, true)
    @test f3([1 0 1; 0 0 1; 1 1 1]) == Bool[1, 0, 1]
    @test g2(Bool[1 1; 0 1]) == Bool[1, 0]
    @test_throws DomainError f3(1, 0)
    @test_throws DomainError g2(3, 0)
    @test_throws DomainError g2(-1, 0)
    @test_throws DomainError f3([1 0; 0 1])
    @test_throws DomainError f3([2 0 1])

    # logicCount and nonZero.
    @test logicCount(f3) == 3 && logicCount(g2) == 1
    @test nonZero(g2) == Bool[1 1]
    @test nonZero(f3; head=2) == Bool[1 0 1; 0 1 1]
    @test nonZero(f3; head=10) == Bool[1 0 1; 0 1 1; 1 1 1]
    @test nonZero(@bfunc(x1 * ~x1)) === nothing
    @test get_non_zero_inputs(g2.val, 2; num=5) == Bool[1 1]
    @test get_non_zero_inputs(g2.val, 2; num=0) == BitMatrix(undef, 0, 2)
    @test_throws DimensionMismatch get_non_zero_inputs(g2.val, 3)
    # The rows are inputs that make the function true.
    nz = nonZero(f3; head=3)
    @test all(f3(Int.(nz[i, :])...) for i in 1:size(nz, 1))
end


@testset "Boolean (Validation and safety) " begin
    init_logic(0)
    # Formulas are data, not code.
    @test_throws ArgumentError Blogic("x1 + exit(3)")
    @test_throws ArgumentError Blogic("x1 + sleep(4)")
    @test_throws ArgumentError Blogic("x1 - x2")
    @test_throws ArgumentError Blogic("x1 + 2")
    @test_throws ArgumentError Blogic("x1 + y2")
    @test_throws ArgumentError Blogic("1 + 0")
    @test_throws ArgumentError Blogic("x1 + (x2")
    @test_throws ArgumentError parseLogic("f(x1)")
    @test_throws ArgumentError Blogic("x1 + x2", "x", BitVector([1, 1, 1]))
    @test_throws DomainError Blogic("x1 + x23")

    # `==` and `hash` agree.
    h1 = create_bool_rep("x1 + x2")
    h2 = create_bool_rep("x1 + x2")
    @test h1 == h2 && hash(h1) == hash(h2) && length(Set([h1, h2])) == 1
    @test h1 != create_bool_rep("x1 * x2")

    # No type piracy: sorting mixed Int/Symbol vectors and printing BitMatrix are Base's business.
    @test_throws MethodError sort(Any[:b, 1, :a])
    @test repr(trues(2, 2)) == "Bool[1 1; 1 1]"

    # show: compact and verbose forms.
    @test repr(h1) == "Blogic(\"x1 + x2\")"
    @test repr("text/plain", h1) == "Formula    = x1 + x2\nVariable   = x\nSize       = 2\nBit vector = Bool[0, 1, 1, 1]"

    # modifyLogicExpr! produces a Julia expression that evaluates to the truth table.
    e = modifyLogicExpr!(parseLogic("x1 * x2"))
    @test e.args[1] == :.& && eval(e) == Bool[0, 0, 0, 1]

    # Randomised check: simplification never changes a truth table.
    rng = Random.MersenneTwister(1)
    function rformula(d)
        d == 0 && return rand(rng, ["x1", "x2", "x3", "0", "1"])
        op = rand(rng, ["+", "*", "⊕", "~", "⟹", "⟺"])
        op == "~" && return "~(" * rformula(d - 1) * ")"
        return "(" * rformula(d - 1) * " " * op * " " * rformula(d - 1) * ")"
    end
    nchecked = 0
    for _ in 1:300
        s = rformula(3)
        occursin(r"[a-z]", s) || continue
        a = create_bool_rep(s; nvars=3)
        b = create_bool_rep(s; simplify=true, nvars=3)
        @test a.val == b.val
        nchecked += 1
    end
    @test nchecked > 200
end
