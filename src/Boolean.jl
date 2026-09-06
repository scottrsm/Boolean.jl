module Boolean

import Base
export Blogic, logicCount, nonZero, get_non_zero_inputs, bool_var_rep
export init_logic, modifyLogicExpr!, simplifyLogic, create_bool_rep
export isEquiv, parseLogic, @bfunc, Blogic_from_file


#=-----------------------------------------------------------------
----------  Module constants   ------------------------------------
-------------------------------------------------------------------
=#

# The largest number of variables a truth table may have (2^22 rows).
const MAX_VARS = 22

# Map from the formula operators to the (broadcast) Julia operators used to evaluate them.
const opMap = Dict(:* => :.&, :+ => :.|, :⊕ => :.⊻, :~ => :.~)

# The operators that may appear in a (parsed) formula.
const LOGIC_OPS = (:*, :+, :⊕, :~)

# Regular expression matching a logic variable: a base name followed by an index.
const VAR_RE = r"^([a-zA-Z]+)(\d+)$"

#= The default minimum number of variables used for truth tables built from formulas
   (see `init_logic`). A formula always uses at least as many variables as its
   highest variable index.
=#
const DEFAULT_LOGIC_SIZE = Ref(0)


"""
    validate_single_variable(s)

Check that a logic formula string uses only one base variable name.
Returns the base variable name string.

# Arguments
- `s :: String` -- A logic formula string.

# Return
`::String` -- The base variable name.
"""
function validate_single_variable(s::AbstractString)
    ar = String[]
    for m in eachmatch(r"([a-zA-Z]+)[0-9]+", s)
        push!(ar, String(m.captures[1]))
    end
    ar = unique(ar)
    if length(ar) > 1
        throw(ArgumentError("Logic string uses more than one variable: $(ar)"))
    end
    if length(ar) == 0
        throw(ArgumentError("Logic string contains no variables."))
    end
    return ar[1]
end

# The highest variable index used in a logic formula string (0 if none).
function max_var_index(s::AbstractString)
    n = 0
    for m in eachmatch(r"[a-zA-Z]+([0-9]+)", s)
        n = max(n, parse(Int, m.captures[1]))
    end
    return n
end


"""
Define an operations type. Meant for the operator symbols:
`:+`, `:*`, `:⊕`, `:~`, so that we may `dispatch` 
on them as `types`.
"""
struct Op{T} end

"""
Structure used to represent a boolean formula involving variables 
given by a single base string followed by a number.

**Note:** The formula to be represented must only contain the 
operators: 
- `~`  -- The NOT operator.
- `*`  -- The AND operator.
- `+`  -- The OR operator.
- `⊕`  -- The XOR operator.
- `⟹ ` -- The implication operator.
- `⟺ ` -- The equivalence operator.

along with variables (`x1`, `x2`, ...; any base name, one per formula),
the constants `0` and `1`, and parentheses. Nothing else is accepted.

The first 4 operators are left associative while the last two are right 
associative. The operator precedence from highest to lowest is:
- `~`
- `*`
- `+`, `xor`
- `⟹ `, `⟺ `

In practice, one uses a higher level constructor (`create_bool_rep`) 
or uses the macro @bfunc. Both of which, in turn, use the inner constructor.

A `Blogic` is self contained: its truth table is stored in `val`, whose
length is ``2^{\\rm size}``. Row `i` of the truth table (entry `val[i]`) is the
value of the formula for the inputs given by the bits of `i - 1`
(variable 1 is the least significant bit). The number of variables of a
formula is the largest variable index it uses, or the minimum set with
`init_logic`, whichever is larger.

# Fields
- `formula :: String`    -- The string representation of the formula.
- `var     :: String`    -- The base name of the logical variables.
- `size    :: Int`       -- The number of variables of the truth table.
- `val     :: BitVector` -- The bit vector representing the formula. 
                            It essentially expresses the values of all 
							possible inputs.  
# Constructors
`Blogic(form::String, v::String, value::BitVector)`

# Examples
```jdoctest
julia> Blogic("(z1 + z2) * z3", "z", BitVector([0, 0, 0, 0, 0, 1, 1, 1]))

Formula    = (z1 + z2) * z3
Variable   = z
Size       = 3
Bit vector = Bool[0, 0, 0, 0, 0, 1, 1, 1]
```

This is the logic (boolean) formula that ORs `z1` and `z2`, 
        then ANDs that with `z3`.
"""
struct Blogic
    formula::String
    var::String
    size::Int
    val::BitVector

    # Inner Constructor
    function Blogic(form::String, v::String, value::BitVector)
        n = length(value)
        (n > 0 && ispow2(n)) || throw(ArgumentError("Blogic: The truth table must have a length that is a power of 2; got $n."))
        return new(form, v, trailing_zeros(n), value)
    end
end


# Clean up a formula string: newlines become spaces, surrounding white space is removed.
clean_formula(s::AbstractString) = strip(replace(s, '\n' => ' ', '\r' => ' '))

# The number of variables to use for a formula string.
function formula_size(s::AbstractString, nvars::Int)
    n = max(max_var_index(s), nvars, DEFAULT_LOGIC_SIZE[])
    1 <= n <= MAX_VARS || throw(DomainError(n, "Blogic: The number of variables must be in the range [1, $MAX_VARS]."))
    return n
end


"""
	Blogic(s[; simplify=false, nvars=0])

Outer constructor for Blogic.

# Arguments
- `s :: String`  -- A logic formula string.

# Keyword Arguments
- `simplify=false::Bool` -- If `true`, the logic is simplified before evaluation.
- `nvars=0::Int` -- The minimum number of variables of the truth table. The number of
                    variables used is the largest of `nvars`, the highest variable index in
                    the formula, and the value set with `init_logic`.

# Return
`::Blogic`
"""
function Blogic(s::AbstractString; simplify::Bool=false, nvars::Int=0)
    s = String(clean_formula(s))
    varname = validate_single_variable(s)
    n = formula_size(s, nvars)
    e = parseLogic(s)
    simplify && (e = simplifyLogic(e))
    value = evaluate_logic(e, n)

    return (Blogic(s, varname, value))
end


"""
	Blogic_from_file(f[; simplify=false, nvars=0])

Outer constructor for Blogic.

# Arguments
- `f :: String`  -- A string representing a utf-8 text file containing a logic formula.

# Keyword Arguments
- `simplify=false::Bool` -- If `true`, the logic is simplified before evaluation.
- `nvars=0::Int` -- The minimum number of variables of the truth table (see `Blogic`).

# Return
`::Blogic`
"""
function Blogic_from_file(f::AbstractString; simplify::Bool=false, nvars::Int=0)
    isfile(f) || throw(ArgumentError("Blogic_from_file: Unable to open file, \"$f\""))
    s = read(f, String)

    return (Blogic(s; simplify=simplify, nvars=nvars))
end


# Check that the inputs of a `Blogic` function are boolean (0/1) values.
function check_bool_inputs(xs)
    all(x -> x == 0 || x == 1, xs) || throw(DomainError(xs, "Blogic function: Inputs must be 0 or 1."))
    return nothing
end

"""
	(Blogic)(xs::Vararg{Integer})

Uses the structure `Blogic` as a `Boolean` function. 

# Arguments
- `xs :: Vararg{Integer}`  -- A Varargs structure representing inputs (`0`/`1` or `Bool`) to the
                              `Blogic` function, `f`; one per variable of `f`.

# Return
`::Bool`
"""
function (f::Blogic)(xs::Vararg{Integer})
    if length(xs) != f.size
        throw(DomainError(length(xs), "Blogic function: Input `xs` has the wrong number of variables (expected $(f.size))."))
    end
    check_bool_inputs(xs)
    p = 1
    s = 0
    for x in xs
        s += Int(x) * p
        p *= 2
    end
    return (f.val[s + 1])
end


"""
	(Blogic)(xm::AbstractMatrix{<:Integer})

Uses the structure `Blogic` as a `Boolean` function. 

# Arguments
- `xm :: AbstractMatrix{<:Integer}`  -- A matrix of size `M`, `N` representing `M` sets of inputs
                                        (`0`/`1` or `Bool`) to the function, `f`, which takes `N` variables.

# Return
`::BitVector` of length `M`.
"""
function (f::Blogic)(xm::AbstractMatrix{<:Integer})
    M, N = size(xm)
    if N != f.size
        throw(DomainError(N, "Blogic function: Input `xm` has the wrong number of variables (expected $(f.size))."))
    end
    check_bool_inputs(xm)

    s = zeros(Int, M)
    p = 1

    for j in 1:N
        @views s .+= xm[:, j] .* p
        p *= 2
    end
    return (f.val[s .+ 1])
end


"""
    create_bool_rep(s; simplify=false, nvars=0)

Turn boolean formula into a `BitVector` representation, `Blogic`.

This is done by the following procedure:
- Determine the underlying base variable used in the formula.
- Parse the formula into an expression, `Expr`.
- Optionally simplify the logical expression.
- Walk the expression tree evaluating it over the `BitVector`
    representations of the variables to create the truth table `BitVector`.

# Arguments 
- `s :: String`      -- A logical string.

# Keyword Arguments
- `simplify=false :: Bool` -- If `true` simplify the logical expression before 
                        creating the `BitVector`.
- `nvars=0::Int` -- The minimum number of variables of the truth table (see `Blogic`).

# Examples
```jdoctest
julia> create_bool_rep("(z1 + z2) * z3")

Formula    = (z1 + z2) * z3
Variable   = z
Size       = 3
Bit vector = Bool[0, 0, 0, 0, 0, 1, 1, 1]
```

# Return
`::Blogic` -- Type representing the logical expression.
"""
create_bool_rep(s::AbstractString; simplify::Bool=false, nvars::Int=0) = Blogic(s; simplify=simplify, nvars=nvars)


#-------------------------------------------------------------------
#----------   The Main Function Interface  -------------------------
#-------------------------------------------------------------------


"""
	@bfunc(x)

A macro to create a `Blogic` function in a syntactically clean way.
This macro determines if an input expression is a valid formula
and creates the associated "truth table" BitVector based on the number of variables
in the formula (see `Blogic`).
Multi-line formulas are entered using a begin/end block. However, each line 
must be a parse-able expression. So, to connect complicated logic use
binary operators on a line by themselves. See the example below.

# Examples
```jdoctest
julia> @bfunc (z1 + z2) * z3

Formula    = (z1 + z2) * z3
Variable   = z
Size       = 3
Bit vector = Bool[0, 0, 0, 0, 0, 1, 1, 1]
```

```jdoctest
julia> @bfunc begin
   (z1 + z2) * z3
   ⟹
   z4 + z5
   end

Formula    = (z1 + z2) * z3 ⟹  z4 + z5
Variable   = z
Size       = 5
Bit vector = Bool[1, 1, 1, 1, 1, 0, 0, 0, 1, 1  …  1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

"""
macro bfunc(x)
    sform = string(x)
    sform = replace(sform, [r"#=.*=#" => "", "begin" => "", "end" => "", "\n" => " ",
        r" *\+ *" => " + ", r" *\* *" => " * ", r" *⟹  *" => " ⟹  ",
        r" *⊕ *" => " ⊕ ", r" *⟺  *" => " ⟺  "]...)

    sform = replace(sform, [r"^ *" => "", r" *$" => "", r" +" => "  "]...)

    # Validate at macro expansion time (so a bad formula is a load-time error),
    # but build the `Blogic` at run time: no global state is touched.
    validate_single_variable(sform)
    parseLogic(sform)

    return :(Blogic($sform))
end


"""
    isEquiv(f1::Blogic, f2::Blogic)

Determines if two logical functions are equivalent when represented as `Blogic` structures.
Functions with different numbers of variables are compared over the larger set of variables.

# Arguments
- `f1 :: Blogic` -- Formula 1.
- `f2 :: Blogic` -- Formula 2.

# Return
`::Bool` -- `true` if the formulas are equivalent; `false` otherwise.

"""
function isEquiv(f1::Blogic, f2::Blogic)
    n = max(f1.size, f2.size)
    return extend_truth_table(f1.val, f1.size, n) == extend_truth_table(f2.val, f2.size, n)
end

#= The truth table of a formula of `n` variables, viewed as a formula of `m >= n` variables:
   the extra (higher) variables do not affect the value, and since variable 1 is the
   least significant bit, the table simply repeats.
=#
function extend_truth_table(val::BitVector, n::Int, m::Int)
    m >= n || throw(DomainError(m, "Cannot shrink a truth table of $n variables to $m variables."))
    return m == n ? val : repeat(val, 2^(m - n))
end


"""
    isEquiv(f1::String, f2::String)

Determines if two logical functions are equivalent when represented as strings.

# Arguments
- `f1 :: String` -- Formula 1.
- `f2 :: String` -- Formula 2.

# Return
`::Bool` -- `true` if the formulas are equivalent; `false` otherwise.

"""
function isEquiv(f1::AbstractString, f2::AbstractString)
    return isEquiv(Blogic(f1), Blogic(f2))
end



#=-----------------------------------------------------------------
----------  Overload Base functions: show, ==, hash      ----------
-------------------------------------------------------------------
=#

"""
    Base.show(io::IO, x::Blogic)

Show a `Blogic` structure: the compact form is `Blogic("formula")`; the
`text/plain` form (used by the REPL) lists the formula, variable, size, and bit vector.
"""
Base.show(io::IO, x::Blogic) = print(io, "Blogic(", repr(x.formula), ")")

function Base.show(io::IO, ::MIME"text/plain", x::Blogic)
    println(io, "Formula    = ", x.formula)
    println(io, "Variable   = ", x.var)
    println(io, "Size       = ", x.size)
    print(io,   "Bit vector = ", x.val)
end


# Define equality (and a matching hash) for type `Blogic`.
"""
	Base.:(==)

Overload the equality function for structures of `Blogic` type.
"""
function Base.:(==)(b1::Blogic, b2::Blogic)
    (b1.formula == b2.formula) &&
        (b1.var == b2.var) &&
        (b1.size == b2.size) &&
        (b1.val == b2.val)
end

Base.hash(b::Blogic, h::UInt) = hash(b.val, hash(b.size, hash(b.var, hash(b.formula, hash(:Blogic, h)))))


#=-----------------------------------------------------------------
----------  Utility Functions   -----------------------------------
-------------------------------------------------------------------
=#

"""
    logicCount(f)

Count the number of true values possible in a given formula.

# Arguments 
- `f :: Blogic` -- A logic formula

# Return
The number of true values that are possible with this formula.

"""
logicCount(f::Blogic) = count(f.val)


"""
    nonZero(f; head=1)

Get up to `head` inputs that generate true values for a logic function, `f`.

# Arguments
- `f :: Blogic` -- A logic formula.

# Keyword Arguments
- `head=1 :: Int`  -- The maximum number of inputs to consider.
    
# Return
`::Union{BitMatrix, Nothing}` -- A matrix whose rows are up to `head` input values
(one column per variable) that will give the logic function, `f`, a value of `true`;
`nothing` if there are none.
"""
function nonZero(f::Blogic; head::Int=1)
    get_non_zero_inputs(f.val, f.size, num=head)
end


"""
	get_non_zero_inputs(v, n[; num=1])

Get up to `num` inputs that generate true values for a logic function.
`v` is a boolean vector that indicates which elements of the truth table
yield a value of `true`.

# Arguments
- `v   :: BitVector` -- A bit vector representing `true` and `false` values.
- `n   :: Int`     -- Describes the length of the truth table column: ``2^n``.

# Keyword Arguments
- `num :: Int`     -- The desired (maximum) number of inputs that generate truth values.

# Return
`::Union{BitMatrix, Nothing}` -- Input values (one row per input, one column per variable)
that generate truth values for the function; `nothing` if there are none.
"""
function get_non_zero_inputs(v::BitVector, n::Int; num::Int=1)
    length(v) == 2^n || throw(DimensionMismatch("get_non_zero_inputs: The bit vector has length $(length(v)); expected 2^$n."))
    num >= 0 || throw(DomainError(num, "get_non_zero_inputs: `num` must be non-negative."))
    idx = findall(v)
    length(idx) == 0 && return (nothing)
    idx = idx[1:min(num, length(idx))]
    return (BitMatrix([((i - 1) >> (j - 1)) & 1 == 1 for i in idx, j in 1:n]))
end


"""
    bool_var_rep(n)

Generate the boolean bit vectors necessary to represent a logic 
formula of `n` variables. 

Essentially, generate the truth table 
of each of the variables collectively as a `BitArray`.

# Arguments
- `n : Number` of logical variables (`1 <= n <= 22`).

# Return
`::BitMatrix` -- The bit representation of all of the logical variables:
a matrix of shape `(2^n, n)`, where column `j` represents variable `j`.
"""
function bool_var_rep(n::Integer)
    1 <= n <= MAX_VARS || throw(DomainError(n, "Can't represent less than 1 or more than $MAX_VARS variables."))
    return BitMatrix([((i - 1) >> (j - 1)) & 1 == 1 for i in 1:2^n, j in 1:n])
end

# The truth table column of variable `j` out of `n` (column `j` of `bool_var_rep(n)`).
function bool_var_column(n::Int, j::Int)
    1 <= j <= n || throw(DomainError(j, "Variable index $j is not in the range [1, $n]."))
    return BitVector([((i - 1) >> (j - 1)) & 1 == 1 for i in 1:2^n])
end


"""
    init_logic(n)

Sets the minimum number of variables, `n`, used for the truth tables of formulas
built from strings or with `@bfunc` (a formula always uses at least as many
variables as its highest variable index). By default there is no minimum.

# Arguments
- `n :: Int` -- The minimum number of boolean variables used for formulas (`0 <= n <= 22`).

# Return
Nothing

"""
function init_logic(n::Integer)
    0 <= n <= MAX_VARS || throw(DomainError(n, "init_logic: `n` must be in the range [0, $MAX_VARS]."))
    DEFAULT_LOGIC_SIZE[] = Int(n)
    return nothing
end



"""
	parseLogic

This function creates a parse tree for a boolean expression.
The Julia function Meta.parse does this -- for the most part.
One of the nice features of this parser is that it collapses
expressions like (+ (x1 (+ x2 (+x3 x4)))) to (+ x1 x2 x3 x4).
It does the same for '*'. However, it does not do so for 
'xor'. We adjust this parse tree from Meta.parse so that it 
does have this property for 'xor'.
We also handle the implication operator and logical equivalence 
operator, by replacing them with their equivalents in terms
of ~, +, or *.

The resulting tree is validated: only the operators `~`, `*`, `+`, `⊕`,
variables of the form `r"[a-zA-Z]+[0-9]+"`, and the constants `0` and `1`
may appear; anything else raises an `ArgumentError`.

# Arguments
- `expr::String` -- A logic formula

# Return
A parse tree with variable string names replaced with symbols.
"""
function parseLogic(expr::AbstractString)

    # Get a parsing with Meta.parse.
    e0 = Meta.parse(String(expr))
    e0 isa Expr && e0.head == :incomplete && throw(ArgumentError("parseLogic: Incomplete logic formula: $(repr(expr))"))

    # Replace logical equivalence operators with the implication operator.
    e1 = fixIffParseTree(e0)

    # Replace implication operators with *, +, and ~.
    e2 = fixImpParseTree(e1)

    # Lastly flatten XOR trees into a vector in the same way that
    # Meta.parse does for + and *.
    e3 = fixXorParseTree(e2)

    # Make sure the tree is a pure logic formula.
    validate_logic_tree(e3)
    return e3
end

# Validate a parsed logic tree: operators, variables, and the constants 0/1 only.
validate_logic_tree(e::Symbol) = (match(VAR_RE, String(e)) === nothing && throw(ArgumentError("parseLogic: Invalid variable name: $e")); nothing)
validate_logic_tree(e::Integer) = ((e == 0 || e == 1) || throw(ArgumentError("parseLogic: Invalid constant: $e (only 0 and 1 are allowed)")); nothing)
validate_logic_tree(e) = throw(ArgumentError("parseLogic: Invalid element in logic formula: $(repr(e))"))
function validate_logic_tree(e::Expr)
    e.head == :call || throw(ArgumentError("parseLogic: Invalid expression in logic formula: $e"))
    op = e.args[1]
    op in LOGIC_OPS || throw(ArgumentError("parseLogic: Invalid operator in logic formula: $op"))
    if op == :~
        length(e.args) == 2 || throw(ArgumentError("parseLogic: `~` takes exactly one argument: $e"))
    else
        length(e.args) >= 3 || throw(ArgumentError("parseLogic: `$op` needs at least two arguments: $e"))
    end
    for a in e.args[2:end]
        validate_logic_tree(a)
    end
    return nothing
end

#= The intent of this function is to "flatten" the parsing from 
   Meta.parse with respect to the "XOR" operator.
   The function is overloaded for three types: Int, Symbol, and Expr.
=#
fixXorParseTree(s) = s

function fixXorParseTree(e::Expr)
    if e.head == :call && e.args[1] == :⊕
        nargs2 = fixXorParseTree(e.args[2])
        nargs3 = fixXorParseTree(e.args[3])
        if nargs2 isa Expr && nargs2.args[1] == :⊕
            nargs2 = nargs2.args[2:end]
        end
        if nargs3 isa Expr && nargs3.args[1] == :⊕
            nargs3 = nargs3.args[2:end]
        end
        return (Expr(:call, :⊕, [nargs2; nargs3]...))
    end
    return (Expr(e.head, map(fixXorParseTree, e.args)...))
end

#= The intent of this function is to replace the logic implication operator
   with its equivalent in terms of NOT and OR: x1 => x2 == ~x1 + x2.
   Again, the function is overloaded for three types: Int, Symbol, and Expr.
=#
fixImpParseTree(s) = s

function fixImpParseTree(e::Expr)
    if e.head == :call && e.args[1] == :⟹
        nargs2 = Expr(:call, :~, fixImpParseTree(e.args[2]))
        nargs3 = fixImpParseTree(e.args[3])
        return (Expr(:call, :+, nargs2, nargs3))
    end
    return (Expr(e.head, map(fixImpParseTree, e.args)...))
end


#= The intent of this function is to replace the logic equivalence operator, ⟺ ,
   with its equivalent in terms of more basic logical operators.
   Again, the function is overloaded for three types: Int, Symbol, and Expr.
   At the leaves of the tree: constants and symbols, we return them unchanged.
   For any expression where the operator is used, we 
   replace it in terms of xor and not:
   x1 ⟺  x2 is the same as: ~x1 ⊕ x2.
=#
fixIffParseTree(s) = s

function fixIffParseTree(e::Expr)
    if e.head == :call && e.args[1] == :⟺
        nargs2 = fixIffParseTree(e.args[2])
        nargs3 = fixIffParseTree(e.args[3])
        return Expr(:call, :⊕, Expr(:call, :~, nargs2), nargs3)
    end
    return (Expr(e.head, map(fixIffParseTree, e.args)...))
end


#=-----------------------------------------------------------------
----------  Evaluation of a logic tree   --------------------------
-------------------------------------------------------------------
=#

"""
    evaluate_logic(e, n)

Evaluate a (parsed) logic expression tree over the truth tables of `n` variables,
producing the truth table (a `BitVector` of length ``2^n``) of the formula.
Constants evaluate to all-false / all-true vectors.
"""
evaluate_logic(e::Integer, n::Int) = (validate_logic_tree(e); e == 1 ? trues(2^n) : falses(2^n))

function evaluate_logic(e::Symbol, n::Int)
    validate_logic_tree(e)
    m = match(VAR_RE, String(e))
    return bool_var_column(n, parse(Int, m.captures[2]))
end

function evaluate_logic(e::Expr, n::Int)
    validate_logic_tree(e)
    op = e.args[1]
    if op == :~
        return .~evaluate_logic(e.args[2], n)
    end
    acc = evaluate_logic(e.args[2], n)
    for a in e.args[3:end]
        b = evaluate_logic(a, n)
        if op == :*
            acc = acc .& b
        elseif op == :+
            acc = acc .| b
        else # :⊕
            acc = acc .⊻ b
        end
    end
    return acc
end


"""
    modifyLogicExpr!(e[, n])

Walk an expression tree, converting the logic operators to the (broadcast) Julia
operators and variables into their `BitVector` truth table representations
(for `n` variables; by default the highest variable index in the expression, or the
minimum set with `init_logic`). The result is a Julia expression that evaluates
to the truth table of the formula. (The package itself evaluates formulas
directly, see `evaluate_logic`; this function is provided for inspection.)

# Arguments
- `e :: Expr` -- A (parsed) logic expression.

# Return
`::Expr` -- A Julia expression over `BitVector`s.
"""
function modifyLogicExpr!(e, n::Int=max(expr_max_var(e), DEFAULT_LOGIC_SIZE[]))
    validate_logic_tree(e)
    return _modify_logic_expr(e, n)
end

_modify_logic_expr(e::Integer, n::Int) = e == 1 ? trues(2^n) : falses(2^n)
function _modify_logic_expr(e::Symbol, n::Int)
    m = match(VAR_RE, String(e))
    return bool_var_column(n, parse(Int, m.captures[2]))
end
function _modify_logic_expr(e::Expr, n::Int)
    return Expr(:call, opMap[e.args[1]], map(a -> _modify_logic_expr(a, n), e.args[2:end])...)
end

# The highest variable index in a logic expression tree.
expr_max_var(e::Integer) = 0
expr_max_var(e::Symbol) = (m = match(VAR_RE, String(e)); m === nothing ? 0 : parse(Int, m.captures[2]))
expr_max_var(e::Expr) = maximum(expr_max_var, e.args[2:end]; init=0)


#=-----------------------------------------------------------------
----------  Simplification of a logic tree   ----------------------
-------------------------------------------------------------------
=#

#= A total order on the elements of a logic tree (Int < Symbol < Expr), used to
   sort the arguments of commutative operators so that equal arguments are adjacent.
   (A private order: `Base.isless` is not extended for `Int`/`Symbol`/`Expr`.)
=#
_rank(::Integer) = 0
_rank(::Symbol) = 1
_rank(::Expr) = 2
_rank(::Any) = 3

function expr_lt(a, b)
    ra, rb = _rank(a), _rank(b)
    ra != rb && return ra < rb
    return _expr_lt_same(a, b)
end
_expr_lt_same(a::Integer, b::Integer) = a < b
_expr_lt_same(a::Symbol, b::Symbol) = a < b
_expr_lt_same(a, b) = false
function _expr_lt_same(a::Expr, b::Expr)
    a.args[1] != b.args[1] && return a.args[1] < b.args[1]
    na, nb = length(a.args), length(b.args)
    na != nb && return na < nb
    for i in 2:na
        expr_lt(a.args[i], b.args[i]) && return true
        expr_lt(b.args[i], a.args[i]) && return false
    end
    return false
end


"""
    rle(xs)

Performs a R(un) L(ength) E(ncoding) on an array, 
grouping like values into arrays.

The values are **assumed** to be sorted.
    
# Arguments
- `xs :: Vector{T}` -- An array that is sortable.

# Return
`::Vector{Tuple{T, Int}}` -- A Vector of pairs of the form: `(T, Int)`
representing values from `xs` and the number of their occurrences.

"""
function rle(xs::Vector{T}) where {T}
    rle = Tuple{T,Int}[]
    isempty(xs) && return rle
    lastx = xs[1]
    cnt = 1
    for x in xs[2:end]
        if x == lastx
            cnt += 1
        else
            push!(rle, (lastx, cnt))
            lastx = x
            cnt = 1
        end
    end
    push!(rle, (lastx, cnt))
    return (rle)
end


"""
    redux(::Op{T}, Tuple{S, Int})

Reduce a pair consisting of an expression and its count to just 
an expression. 

The default case is to just return the expression.

# Arguments
- `::Op{T}`                    -- An operator type.
- `pair :: Tuple{Expr, Int}` -- Expression and its count.

# Return
`::Expr` -- Simplified logic expression.
"""
function redux(::Op{T}, pair::Tuple{S,Int}) where {S,T}
    return (pair[1])
end


"""
    redux(::Op{:⊕}, pair::Tuple{Expr, Int})

Reduce a pair consisting of an expression and its count to just 
an expression. 

For an XOR expression, we know that only the expression 
remains or the value is 0.

# Arguments
- `:::Op{:⊕}`                  -- An operator type.
- `pair :: Tuple{Expr, Int}` -- Expression and its count.

# Return
`::Expr` -- Simplified logic expression.
"""
function redux(::Op{:⊕}, pair::Tuple{S,Int}) where {S}
    if pair[2] % 2 == 0
        return (0)
    else
        return (pair[1])
    end
end


"""
    simplifyLogic(e)

Simplify a logical expression.

This function calls a number of specialized variations of this function 
to deal with different logical operators.

# Arguments
- `e :: Expr` -- Logic expression.

# Return
`::Expr` -- Simplified logic expression (an `Int` if it reduces to a constant, a `Symbol` if to a variable).

"""
function simplifyLogic(e::Expr)
    if length(e.args) >= 3
        op = e.args[1]
        return (simplifyLogic(Op{op}(), Any[e.args[2:end]...]))
    end
    # If this has the form: `~ expr...`
    if length(e.args) == 2 && e.args[1] == :~
        if e.args[2] isa Expr && length(e.args[2].args) == 2 && e.args[2].args[1] == :~
            return (simplifyLogic(e.args[2].args[2]))
        end
        arg = simplifyLogic(e.args[2])
        if arg isa Integer
            return ((1 + arg) % 2)
        else
            return (Expr(:call, :~, arg))
        end
    end

    return (e)
end

# Simplify, sort, and merge repeated arguments of an n-ary operator.
function _simplified_args(op::Symbol, xargs::Vector{Any})
    xargs = Any[simplifyLogic(arg) for arg in xargs]
    return Any[redux(Op{op}(), x) for x in rle(sort(xargs; lt=expr_lt))]
end


"""
    simplifyLogic(::Op{:+}, xargs::Vector{Any})

`simplifLogic` for the OR operator.
"""
function simplifyLogic(::Op{:+}, xargs::Vector{Any})
    xargs = _simplified_args(:+, xargs)

    if any(x -> x == 1, xargs)
        return (1)
    end
    xargs = filter(x -> x != 0, xargs)
    if length(xargs) == 0
        return (0)
    elseif length(xargs) == 1
        return (xargs[1])
    else
        return (Expr(:call, :+, xargs...))
    end
end


"""
    simplifyLogic(::Op{:*}, xargs::Vector{Any})

`simplifyLogic` for the AND operator.
"""
function simplifyLogic(::Op{:*}, xargs::Vector{Any})
    xargs = _simplified_args(:*, xargs)

    if any(x -> x == 0, xargs)
        return (0)
    end
    xargs = filter(x -> x != 1, xargs)
    if length(xargs) == 0
        return (1)
    elseif length(xargs) == 1
        return (xargs[1])
    else
        return (Expr(:call, :*, xargs...))
    end
end


"""
    simplifyLogic(::Op{:⊕}, xargs::Vector{Any})

`simplifyLogic` for the XOR operator.
"""
function simplifyLogic(::Op{:⊕}, xargs::Vector{Any})
    xargs = _simplified_args(:⊕, xargs)

    iargs = filter(arg -> arg isa Integer, xargs)
    xargs = filter(arg -> !(arg isa Integer), xargs)
    # If there are no simple booleans (0 or 1s), return the xor expression 
    #      with the xargs.
    if length(iargs) == 0
        if length(xargs) == 1
            return (xargs[1])
        end
        return (Expr(:call, :⊕, xargs...))
    end

    # If there are no complex boolean expressions, return the xor 
    #      value of the simple booleans.
    if length(xargs) == 0
        return (sum(iargs) % 2)
        #= else if there is one complex boolean expression, return the 
           expression that is the xor of the resulting simple boolean XORS 
           with the complex boolean expression.
		=#
    elseif length(xargs) == 1
        if (sum(iargs) % 2) == 1
            return (Expr(:call, :~, xargs[1]))
        else
            return (xargs[1])
        end
    end

    # Otherwise, there is a simple component, find its xor value 
    # and then return an expression of the xor with the complex expressions.
    if (sum(iargs) % 2) == 1
        return (Expr(:call, :~, Expr(:call, :⊕, xargs...)))
    else
        return (Expr(:call, :⊕, xargs...))
    end
end


"""
    simplifyLogic(e::Union{Int, Symbol})

`simplifyLogic` for the irreducible cases: A number or a symbol.
"""
function simplifyLogic(e::Union{Integer,Symbol})
    return e
end



end # module Boolean
