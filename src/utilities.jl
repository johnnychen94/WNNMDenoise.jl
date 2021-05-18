@doc raw"""
    soft_threshold(X, γ)

Soft thresholding `X` with threshold `γ` using `sign(X) * max(abs(X) - γ, 0)`. Broadcasting is applied
when necessarily.

# Examples

`soft_threshold(X, 1/λ)` is the solution to the simplified lasso problem:

```math
    \min_d \lvert d \rvert + \frac{\lambda}{2} \lVert d - X \rVert^2_2
```

# References

[1] Goldstein, T., & Osher, S. (2009). The split Bregman method for L1-regularized problems. _SIAM journal on imaging sciences_, 2(2), 323-343.

[2] Wikipedia contributors. (2020, July 15). Lasso (statistics). In _Wikipedia, The Free Encyclopedia_. Retrieved 10:40, August 21, 2020, from https://en.wikipedia.org/w/index.php?title=Lasso_(statistics)&oldid=967820964

"""
soft_threshold(X, γ) = soft_threshold.(X, γ)
soft_threshold(x::T, γ::Number) where T <: Number = sign(x) * max(abs(x) - _maybe_promote(T, γ), zero(T))
# TODO: for cases that x is positve, max(x - γ, 0) is slightly faster

_maybe_promote(::Type{T}, x) where T = convert(T, x)
_maybe_promote(::Type{T1}, x::T2) where {T1 <: Union{Bool,Integer},T2} = promote_type(T1, T2)(x)

# Threads helper

function get_num_threads()
    blas = LinearAlgebra.BLAS.vendor()
    # Wrap in a try to catch unsupported blas versions
    try
        if blas == :openblas
            return ccall((:openblas_get_num_threads, Base.libblas_name), Cint, ())
        elseif blas == :openblas64
            return ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
        elseif blas == :mkl
            return ccall((:MKL_Get_Max_Num_Threads, Base.libblas_name), Cint, ())
        end

        # OSX BLAS looks at an environment variable
        if Sys.isapple()
            return tryparse(Cint, get(ENV, "VECLIB_MAXIMUM_THREADS", "1"))
        end
    catch
    end

    return nothing
end

function with_blas_threads(f, num_threads)
    prev_num_threads = get_num_threads()
    prev_num_threads = isnothing(prev_num_threads) ? 1 : prev_num_threads
    BLAS.set_num_threads(num_threads)
    retval = nothing
    try
        retval = f()
    catch err
        rethrow(err)
    finally
        BLAS.set_num_threads(prev_num_threads)
    end

    return retval
end
