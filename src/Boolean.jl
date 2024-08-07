module Boolean

global vars = nothing
global logic_size = nothing
global opMap = Dict(:* => :.&, :+ => :.|, :⊕ => :.⊻, :~ => :.~)

import Base
export Blogic, logicCount, nonZero, get_non_zero_inputs, bool_var_rep
export init_logic, modifyLogicExpr!, simplifyLogic, create_bool_rep
export isEquiv, parseLogic, @bfunc, Blogic_from_file


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

The first 4 operators are left associative while the last two are right 
associative. The operator precedence from highest to lowest is:
- `~`
- `*`
- `+`, `xor`
- `⟹ `, `⟺ `

In practice, one uses a higher level constructor (`create_bool_rep`) 
or uses the macro @bfunc. Both of which, in turn, use the inner constructor.

# Fields
- `formula :: String`    -- The string representation of the formula.
- `var     :: String`    -- The base name of the logical variables.
- `size    :: Int`       -- The number of variables in the formula.
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
    Blogic(form::String, v::String, value::BitVector) =
        new(form, v, Int(log2(length(value))), value)
end


"""
	Blogic(s[; simplify])

Outer constructor for Blogic.

# Arguments
- `s :: String`  -- A logic formula string.

# Optional Arguments
- `simplify=false::Bool` - Bool argument, if `true`, logic should be simplified.

# Return
`::Blogic`
"""
function Blogic(s::String; simplify::Bool=false)

    s = replace(s, '\n' => ' ')
    value = eval(create_boolean_expr_tree(s; simplify=simplify))

    #= Check that the variables used have the same name:
       Looking for x1, x2, x3. Not x1, y2, z3.
       Get the array of unique variables names.
	=#
    ar = []
    for m in eachmatch(r"[a-zA-Z]+([0-9]+)", s)
        push!(ar, split(m.match, r"[0-9]+")[1])
    end

    # If there are more than one, error.
    ar = unique(ar)
    if length(ar) > 1
        error("Logic string uses more than one variable: ",
            map(x -> String(x), ar))
    end

    return (Blogic(s, String(ar[1]), value))
end


"""
	Blogic_from_file(f[; simplify])

Outer constructor for Blogic.

# Arguments
- `f :: String`  -- A string representing a utf-8 text file containing a logic formula.

# Return
`::Blogic`
"""
function Blogic_from_file(f::String; simplify::Bool=false)

    fh = nothing
    try
        fh = open(f)
    catch
        throw(DomainError(0, "Blogic: Unable to open file, \"$f\""))
    end

    s = read(fh, String)
    close(fh)

    return (Blogic(s; simplify=simplify))
end


"""
	(Blogic)(xs::Vararg{Int})

Uses the structure `Blogic` as a `Boolean` function. 

# Arguments
- `xm :: Vararg{Int}`  -- A Varargs structure representing inputs to the
                           `Blogic` function, `f`.

# Return
`::Bool`
"""
function (f::Blogic)(xs::Vararg{Int})
    global logic_size

    if length(xs) != logic_size
        raise(DomainError("Blogic function: Input `xs` has the wrong number of variables."))
    end
    p = 1
    s = 0
    for x in xs
        s += x * p
        p *= 2
    end
    return (f.val[s])
end


"""
	(Blogic)(xm::Matrix{Int})

Uses the structure `Blogic` as a `Boolean` function. 

# Arguments
- `xm :: Matrix{Int}`  -- A matrix of size `M`, `N` representing `M` sets of inputs
                            to the function, `f`, which takes `N` variables.

# Return
`::BitVector` of length `M`.
"""
function (f::Blogic)(xm::Matrix{Int})
    global logic_size

    M, N = size(xm)
    if N != logic_size
        raise(DomainError("Blogic function: Input `xm` has the wrong number of variables."))
    end

    s = zeros(Int, M)
    p = 1

    for i in 1:N
        s += xm[:, i] .* p
        p *= 2
    end
    return (f.val[s])
end


"""
    create_bool_rep(s, simplify=false)

Turn boolean formula into a `BitVector` representation, `Blogic`.

This is done by the following procedure:
- Determine the underlying base variable used in the formula.
- Parse the formula into an expression, `Expr`.
- Optionally simplify the logical expression.
- Walk the expression tree creating a new tree with Julia 
    mathematical operators substituted for user operators.
- Evaluate the expression to create a `BitVector`.

# Arguments 
- `s :: String`      -- A logical string.
- `simplify=false :: Bool` -- If `true` simplify the logical expression before 
                        creating the `BitVector`.
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
function create_bool_rep(s::String, simplify=false)
    global logic_size

    #= Check that the variables used have the same name:
       Looking for x1, x2, x3. Not x1, y2, z3.
       Get the array of unique variables names.
	=#
    ar = []
    for m in eachmatch(r"[a-zA-Z]+([0-9]+)", s)
        push!(ar, split(m.match, r"[0-9]+")[1])
    end

    # If there are more than one, error.
    ar = unique(ar)
    if length(ar) > 1
        error("Logic string uses more than one variable: ",
            map(x -> String(x), ar))
    end
    ns = parseLogic(s)
    if simplify
        val = eval(modifyLogicExpr!(simplifyLogic(ns)))
    else
        val = eval(modifyLogicExpr!(ns))
    end
    Blogic(s, String(ar[1]), val)
end

function create_boolean_expr_tree(s::String; simplify::Bool=false)
    ns = parseLogic(s)
    if simplify
        val = modifyLogicExpr!(simplifyLogic(ns))
    else
        val = modifyLogicExpr!(ns)
    end
    return (val)
end

#-------------------------------------------------------------------
#----------   The Main Function Interface  -------------------------
#-------------------------------------------------------------------


"""
	@bfunc(x)

A macro to create a `Blogic` function in a syntactically clean way.
This macro determes if an input expression is a valid formula
and creates the associated "truth table" BitVectors based on the number of variables
in the formula. The function that does this is `init_logic` which modifies
**global variables**.
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
    num_vars = maximum([parse(Int, x) for x in split(sform, r"[ *+~⟺ ()⊕⟹ a-z]+") if x != ""])

    #= Check that the variables used have the same name:
       Looking for x1, x2, x3. Not x1, y2, z3.
       Get the array of unique variables names.
	=#
    ar = []
    for m in eachmatch(r"[a-zA-Z]+([0-9]+)", sform)
        push!(ar, split(m.match, r"[0-9]+")[1])
    end

    # If there is more than one, error.
    ar = unique(ar)
    if length(ar) > 1
        error("Logic string uses more than one variable: ",
            map(x -> String(x), ar))
    end

    # Build bit-vector versions of `num_vars` logic variables.
    init_logic(num_vars)


    #= Create the call to Blogic inner constructor which will take the string form
       of the formula, the variable base name, and the expression tree
       which will be evaluated to a bit-vector -- our representation of the function.
	=#
    Expr(:call, :Blogic, sform, String(ar[1]), create_boolean_expr_tree(sform))
end


"""
    isEquiv(f1::Blogic, f2::Blogic)

Determines if two logical functions are equivalent when represented as `Blogic` structures.

# Arguments
- `f1 :: Blogic` -- Formula 1.
- `f2 :: Blogic` -- Formula 2.

# Return
`::Bool` -- `true` if the formulas are equivalent; `false` otherwise.

"""
function isEquiv(f1::Blogic, f2::Blogic)
    sform = "( " * f1.formula * " ) ⊕ ( " * f2.formula * " )"
    b = create_bool_rep(sform)
    lc = logicCount(b)
    return (lc == 0 ? true : false)
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
function isEquiv(f1::String, f2::String)
    b = create_bool_rep("( " * f1 * ") ⊕ " * " ( " * f2 * " )")
    lc = logicCount(b)
    return (lc == 0 ? true : false)
end



#=-----------------------------------------------------------------
----------  Overload Base functions: show, isless, ==    ----------
-------------------------------------------------------------------
=#

"""
    Show the Blogic structure.
    Params:
    io: IO handle.
    x : Blogic structure.
"""
function Base.show(io::IO, x::Blogic)
    println(io, "Formula    = ", x.formula)
    println(io, "Variable   = ", x.var)
    println(io, "Size       = ", x.size)
    println(io, "Bit vector = ", x.val)
end


"""
    Show a BitMatrix.
"""
function Base.show(io::IO, z::BitMatrix)
    n, _ = size(z)
    if n == 0
        println("N/A")
    else
        for i in 1:n
            println(io, Tuple(map(x -> Int(x), z[i, :])))
        end
    end
end


# Define how to order Int, Symbols, and Expr.
# Needed for `simplifyLogic`.
"""
	Base.isless(i1::Int, s2::Symbol)

Compare an `Int` with a `Symbol`.
"""
Base.isless(x::Int, y::Symbol) = true
Base.isless(x::Symbol, y::Int) = false

"""
	Base.isless(i1::Int, e2::Expr)

Compare an `Int` with an `Expr`.
"""
Base.isless(x::Int, y::Expr) = true
Base.isless(x::Expr, y::Int) = false

"""
	Base.isless(s1::Symbol, e2::Expr)

Compare an `Symbol` with an `Expr`.
"""
Base.isless(x::Symbol, y::Expr) = true
Base.isless(x::Expr, y::Symbol) = false

"""
	Base.isless(e1::Expr, e2::Expr)

Compare two Julia `Expr` expressions.
"""
function Base.isless(e1::Expr, e2::Expr)
    args1 = e1.args
    args2 = e2.args
    args1[1] < args2[1] && return (true)
    args1[1] > args2[1] && return (false)
    n1 = length(args1[2:end])
    n2 = length(args2[2:end])
    n1 < n2 && return (true)
    n1 > n2 && return (false)
    for i in 1:n1
        Base.isless(args1[1+i], args2[1+i]) && return (true)
    end
    return (false)
end


# Define equality for type `Blogic`.
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
    nonZero(f, head=1)

Get up to `head` inputs that generate true values for a logic function, `f`.

# Arguments
- `f :: Blogic` -- A logic formula.

# Optional Arguments
- `head=1 :: Int`  -- The maximum number of inputs to consider.
    
# Return
A list of up to `head` input values that will give the 
logic function, `f`, a value of `true`.
"""
function nonZero(f::Blogic; head=1)
    n = logicCount(f)
    get_non_zero_inputs(f.val, f.size, num=min(n, head))
end


"""
	get_non_zero_inputs(v, n[; num=1])

Get `num` inputs that generate true values for a logic function.
`v` is a boolean vector that indicates which elements of the truth table
yield a value of `true`.

# Arguments
- `v   :: BitVector` -- A bit vector representing `true` and `false` values.
- `n   :: Int`     -- Describes the length of the truth table column: ``2^n``.

# Optional Arguments
- `num :: Int`     -- The desired number of inputs that generate truth values.

# Returns
`::Union{BitMatrix, Nothing}` -- Input values that generate truth values for the current function.
"""
function get_non_zero_inputs(v::BitVector, n::Int; num::Int=1)
    idx = collect(1:2^n)[v]
    length(idx) == 0 && return (nothing)
    return (vars[idx[1:num], :])
end


"""
    bool_var_rep(n)

Generate the boolean bit vectors necessary to represent a logic 
formula of `n` variables. 

Essentially, generate the truth table 
of each of the variables collectively as a `BitArray`.

# Arguments
- `n : Number` of logical variables.

# Return
`::BitArray` -- The bit representation of all of the logical variables.
"""
function bool_var_rep(n::Signed)
    if n > 30
        error("Can't represent more than 30 variables.")
    elseif n < 2
        error("Can't represent less than 2 variables.")
    else
        let nn = Unsigned(n)
            #= `BitArray([div(i-1, 2^(j-1)) % 2 != 0  for i in 1:2^n, j in 1:n])`
               This is a bit matrix of shape (2^n, n), where column 1 
               represents `x1`, column 2 represents `x2`, etc.
			=#
            BitArray([((i - 1) >> (j - 1)) & 1 for i in 1:2^nn, j in 1:nn])
        end
    end
end


"""
    init_logic

This sets two global variables, the size of the boolean vectors and 
the other the `Bitarray` representations of the variables.

# Arguments
- `n :: Int` -- The number of boolean variables used in the formulas
this module will consider.

# Return
Nothing

"""
function init_logic(n::Signed)
    global vars = bool_var_rep(n)
    global logic_size = n
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

# Arguments
- `expr::String` -- A logic formula

# Return
A parse tree with variable string names replaced with symbols.
"""
function parseLogic(expr::String)

    # Get a parsing with Meta.parse.
    e0 = Meta.parse(expr)

    # Replace logical equivalence operators with the implication operator.
    e1 = fixIffParseTree(e0)

    # Replace implication operators with *, +, and ~.
    e2 = fixImpParseTree(e1)

    # Lastly flatten XOR trees into a vector in the same was that
    # Meta.parse does for + and *.
    return (fixXorParseTree(e2))
end

#= Given a vector of expressions create that interleaves an expression with
   a logic operator -- one of: +, *, xor, not, implies, is-equivalent,
   create an new expression that is the "left associative" parse of this.
   That is, treat all operators as left associative with equal precedence.
=#
function process_multiline_logic(ex::Expr)
    if ex.head != :block
        return (ex)
    end

    op_ary = []
    arg_ary = []
    for e in ex.args
        if e in [:+, :*, :⊕, :⟹, :⟺]
            push!(op_ary, e)
        else
            push!(arg_ary, e)
        end
    end
    if length(op_ary) + 1 != length(arg_ary)
        throw(DomainError(0, "Every other line should be an operator."))
    end
    reverse!(op_ary)
    reverse!(arg_ary)
    e = Expr(:call, pop!(op_ary), pop!(arg_ary), pop!(arg_ary))
    while length(op_ary) > 0
        e = Expr(:call, pop!(op_ary), e, pop!(arg_ary))
    end
    return (e)
end

#= The intent of this function is to "flatten" the parsing from 
   Meta.parse with respect to the "XOR" operator.
   The function is overloaded for three types: Int, Symbol, and Expr.
=#
function fixXorParseTree(s::Int, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


function fixXorParseTree(s::Symbol, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


function fixXorParseTree(e::Expr, cnt=1; verbose=false)
    N = length(e.args)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Expr: $e")

    if e.args[1] == :⊕
        nargs2 = fixXorParseTree(e.args[2], cnt + 1; verbose=verbose)
        nargs3 = fixXorParseTree(e.args[3], cnt + 1; verbose=verbose)
        if typeof(nargs2) == Expr && nargs2.args[1] == :⊕
            nargs2 = deepcopy(nargs2.args[2:end])
        end
        if typeof(nargs3) == Expr && nargs3.args[1] == :⊕
            nargs3 = deepcopy(nargs3.args[2:end])
        end
        return (Expr(:call, :⊕, [nargs2; nargs3]...))
    end
    return (Expr(:call, e.args[1], map(x -> fixXorParseTree(x, cnt + 1; verbose=verbose), e.args[2:end])...))
end

#= The intent of this function is to replace the logic implication operator
   with its equivalent in terms of NOT and OR: x1 => x2 == ~x1 + x2.
   Again, the function is overloaded for three types: Int, Symbol, and Expr.
=#
function fixImpParseTree(s::Int, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


function fixImpParseTree(s::Symbol, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


function fixImpParseTree(e::Expr, cnt=1; verbose=false)
    N = length(e.args)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Expr: $e")

    if e.args[1] == :⟹
        nargs2 = Expr(:call, :~, fixImpParseTree(e.args[2]))
        nargs3 = fixImpParseTree(e.args[3], cnt + 1; verbose=verbose)
        return (Expr(:call, :+, [nargs2; nargs3]...))
    end
    return (Expr(:call, e.args[1], map(x -> fixImpParseTree(x, cnt + 1; verbose=verbose), e.args[2:end])...))
end


#= The intent of this function is to replace the logic equivalence operator, ⟺ ,
   with its equivalent in terms of more basic logical operators.
   Again, the function is overloaded for three types: Int, Symbol, and Expr.
   At the leaves of the tree: constants and symbols, we return them unchanged.
=#
function fixIffParseTree(s::Int, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


function fixIffParseTree(s::Symbol, cnt=1; verbose=false)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Symbol: $s")
    return (s)
end


#= For any expression where the operator is used, we 
   repace ot in terms of xor and not:
   x1 ⟺  x2 is the same as: ~x1 ⊕ x2.
=#
function fixIffParseTree(e::Expr, cnt=1; verbose=false)
    N = length(e.args)
    delim = join(fill("  ", cnt))
    verbose && println("$(delim)Expr: $e")

    if e.args[1] == :⟺
        nargs2 = fixIffParseTree(e.args[2], cnt + 1; verbose=verbose)
        nargs3 = fixIffParseTree(e.args[3], cnt + 1; verbose=verbose)
        exp = Expr(:call, :⊕, Expr(:call, :~, nargs2), nargs3)
        return (exp)
    end
    return (Expr(:call, e.args[1], map(x -> fixIffParseTree(x, cnt + 1; verbose=verbose), e.args[2:end])...))
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
    lastx = xs[1]
    rle = []
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
    modifyLogicExpr!(e)

The default rule for modifying a logic expression is to do nothing.
"""
function modifyLogicExpr!(e::T) where {T}
    return (e)
end


"""
    modifyLogicExpr!(e::Expr)

Walk an expression tree, converting variable names and operators
to Julia operators and variables into `BitVector` representations.

# Arguments
- `e :: Expr` -- An expression.

# Return
`::Expr` -- A logic expression.
"""
function modifyLogicExpr!(e::Expr)
    ary = []
    for (_, arg) in enumerate(e.args)
        push!(ary, modifyLogicExpr!(arg))
    end
    e.args = ary
    return (e)
end


"""
    modifyLogicExpr!(e::Symbol)

If `e` is a Symbol, it should be a variable of the form `r"[a-zA-Z]+[0-9]+"`.

The code splits the name off and uses the number to look up the 
    `BitVector` representation.
    Otherwise, it is assumed to be an operator symbol and it is then 
    mapped to the appropriate Julia operator.

  **NOTE:** This will work even if one makes a mistake and uses 
            `x3`, or `y3`, the bit vector for the 
            third "variable" will be used.

# Arguments
- `e :: Symbol` -- An variable or operator.

# Return
`::Expr` -- A logic expression.
"""
function modifyLogicExpr!(e::Symbol)
    global vars
    global opMap

    # If this is a variable get the corresponding `BitVector`.
    if match(r"[a-zA-Z]+", String(e)) !== nothing
        vn = parse(Int, (split(String(e), r"[a-zA-Z]+"))[2])
        return (vars[:, vn])
    end

    # If this is an operator symbol, get the corresponding Julia operator.
    return (get(opMap, e, e))
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
    redux(::Opt{:⊕}, pair::Tuple{Expr, Int})

Reduce a pair consisting of an expression and its count to just 
an expression. 

For an XOR expression, we know that only the expression 
remains or the value is 0.

# Arguments
- `:::Opt{:⊕}`                 -- An operator type.
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
`::Expr` -- Simplified logic expression.

"""
function simplifyLogic(e::Expr)
    if length(e.args) >= 3
        op = e.args[1]
        return (simplifyLogic(Op{op}(), e.args[2:end]))
    end
    # If this has the form: `~ expr...`
    if length(e.args) == 2 && e.args[1] == :~
        if typeof(e.args[2]) == Expr && length(e.args[2].args) == 2
            e.args[2].args[1] == :~
            return (simplifyLogic(e.args[2].args[2]))
        end
        arg = simplifyLogic(e.args[2])
        if typeof(arg) == Int
            return ((1 + arg) % 2)
        else
            return (Expr(:call, :~, arg))
        end
    end

    return (e)
end


"""
    simplifyLogic(::Op{:~}, xargs::Any)

`simplifyLogic` for the NOT operator.
"""
function simplifyLogic(::Op{:~}, xargs::Any)
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:~}(), x), rle(sort(xargs)))

    if xargs == 1
        return (0)
    end
    if xargs == 0
        return (1)
    end
    return (Expr(:call, :~, xargs))
end


"""
    simplifyLogic(::Op{:+}, xargs::Vector{Any})

`simplifLogic` for the OR operator.
"""
function simplifyLogic(::Op{:+}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:+}(), x), rle(sort(xargs)))

    if any(x -> x == 1, xargs)
        return (1)
    end
    xargs = filter(x -> x != 0, xargs)
    if length(xargs) == 0
        return (0)
    elseif length(xargs) == 1
        if xargs[1] isa Vector{Any}
            return (Expr(xargs[1]...))
        else
            return (xargs[1])
        end
    else
        return (Expr(:call, :+, xargs...))
    end
end


"""
    simplifyLogic(::Op{:*}, xargs::Vector{Any})

`simplifyLogic` for the AND operator.
"""
function simplifyLogic(::Op{:*}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:*}(), x), rle(sort(xargs)))

    if any(x -> x == 0, xargs)
        return (0)
    end
    xargs = filter(x -> x != 1, xargs)
    if length(xargs) == 0
        return (1)
    elseif length(xargs) == 1
        if xargs[1] isa Vector{Any}
            return (Expr(xargs[1]...))
        else
            return (xargs[1])
        end
    else
        return (Expr(:call, :*, xargs...))
    end
end


"""
    simplifyLogic(::Op{:⊕}, xargs::Vector{Any})

`simplifyLogic` for the XOR operator.
"""
function simplifyLogic(::Op{:⊕}, xargs::Vector{Any})
    xargs = map(arg -> simplifyLogic(arg), xargs)
    xargs = map(x -> redux(Op{:⊕}(), x), rle(sort(xargs)))

    iargs = filter(arg -> typeof(arg) == Int, xargs)
    xargs = filter(arg -> typeof(arg) != Int, xargs)
    # If there are no simple booleans (0 or 1s), return the xor expression 
    #      with the xargs.
    if length(iargs) == 0
        return (Expr(:call, :⊕, xargs...))
        if xargs[1] isa Vector{Any}
            return (Expr(xargs[1]...))
        else
            return (xargs[1])
        end
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
            if xargs[1] isa Vector{Any}
                return (Expr(xargs[1]...))
            else
                return (xargs[1])
            end
        end
    end

    # Otherwise, there is a simple component, find its xor value 
    # and then return an expression of the xor with the complex expressions.
    if (sum(iargs) % 2) == 1
        return (Expr(:call, :~, Expr(:call, :⊕, xargs...)))
    else
        return (Expr(:call, :⊕, xargs...))
    end

    # We should not make it here.
    throw(DomainError(0, "Method,simplifyLogic failed."))
end


"""
    simplifyLogic(e::Union{Int, Symbol})

`simplifyLogic` for the irreducible cases: A number or a symbol.
"""
function simplifyLogic(e::Union{Int,Symbol})
    return e
end



end # module Boolean
