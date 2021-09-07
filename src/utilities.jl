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

# optimized for memory
function mean2!(out::AbstractVector, m::AbstractMatrix)
    @assert length(out) == size(m, 1)
    fill!(out, zero(eltype(out)))
    @inbounds for i in axes(m, 1)
        rst = out[i]
        @simd for j in axes(m, 2)
            rst += m[i, j]
        end
        out[i] = rst / size(m, 2)
    end
    return out
end

# Copy from BlockMatching with extra `dist_buffer` input to reuse the memory
function multi_match(S::FullSearch, ref, frame, p::CartesianIndex, dist_buffer; num_patches)
    rₚ = S.patch_radius
    R_frame = CartesianIndices(frame)
    R_frame = first(R_frame) + rₚ : last(R_frame) - rₚ
    p in R_frame || throw(ArgumentError("Boundary unsupported now: pixel `p = $p` should be within $(first(R_frame)):$(last(R_frame))."))

    R_ref = CartesianIndices(ref)
    R_ref = first(R_ref) + rₚ : last(R_ref) - rₚ
    q_start = max(first(R_ref), p - S.search_radius)
    q_stop = min(last(R_ref), p + S.search_radius)
    candidates = q_start:S.search_stride:q_stop

    # For performance, we do not need to compare the dense patch to do block matching
    Δ = CartesianIndex(2, 2)
    patch_p = frame[p - rₚ:Δ:p + rₚ]
    @assert length(dist_buffer) >= length(candidates)
    dist = dist_buffer
    @inbounds for i in 1:length(candidates)
        q = candidates[i]
        patch_q = @view ref[q - rₚ:Δ:q + rₚ]
        dist[i] = S.f(patch_p, patch_q)
    end

    indices = partialsortperm(view(dist, 1:length(candidates)), 1:num_patches)
    candidates[indices]
end

# Threads helper

function with_blas_threads(f, num_threads)
    prev_num_threads = BLAS.get_num_threads()
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
