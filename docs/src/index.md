# Boolean.jl Documentation

```@meta
CurrentModule = Boolean
```
# Overview
This module contains functions to compare Boolean functions.
It does this by using a bit-vector representation and 
comparing bits. The drawback with this representation is that it 
grows exponentially with the number of variables in a boolean expression.

High level constructors take an algebraic formula as the user representation
of the logic function. This is then parsed and converted to a `BitVector` 
representing the output of all possible inputs to its "truth table".

**Note:** When parsing, the operator precedence is respected, with the 
same precedence that Julia obeys. In particular, `xor` has the same precedence as `or`.

There is an associated Jupyter notebook at src/BoolTest.ipynb.

## Types

```@docs
Op
```

```@docs
Blogic
```

## Alternative `Blogic` Constructor

```@docs
Blogic(s::String; simplify::Bool=false)
```

```@docs
Blogic_from_file(f::String; simplify::Bool=false)
```

```@docs
create_bool_rep
```

## High Level Module Functions
```@docs
@bfunc
```

```@docs
(Blogic)(::Vararg{Int})
```

```@docs
(Blogic)(::Matrix{Int})
```

```@docs
isEquiv(f1::String, f2::String)
```

```@docs
isEquiv(f1::Blogic, f2::Blogic)
```

## Base Overloaded Functions
Used to show `Blogic` structures and compare Symbols with Expressions.

```@docs
Base.show(::IO, ::Blogic)
```

```@docs
Base.show(::IO, ::BitMatrix)
```

```@docs
Base.isless(::Int, ::Symbol)
```

```@docs
Base.isless(::Int, ::Expr)
```

```@docs
Base.isless(::Symbol, ::Expr)
```

```@docs
Base.isless(::Expr, ::Expr)
```

```@docs
Base.:(==)(::Blogic, ::Blogic)
```

## Low Level Functions

```@docs
logicCount
```

```@docs
nonZero
```

```@docs
get_non_zero_inputs
```

```@docs
bool_var_rep
```

```@docs
init_logic
```

```@docs
parseLogic
```

```@docs
modifyLogicExpr!
```

```@docs
simplifyLogic
```


```@docs
rle
```

```@docs
redux
```

## Index

```@index
```

