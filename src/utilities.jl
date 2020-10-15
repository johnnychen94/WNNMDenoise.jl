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


"""
    block_matching([f=mse,] img, p; kwargs...)

Given the patch center `p` and image `img`, search the `num_patches` most similar patches to
patch_p. The returned results are `Vector{CartesianIndices}` which stores the positions of each
patch.

# Outputs

* `indices::Vector{<:CartesianIndices}`: each item is the CartesianIndices of the matched patch.

# Arguments

* `f`: a smaller `f(patch, ref_patch)` means these two patches are more similar. The default measure
   is `ImageDistances.mse`.
* `img`: the image that block matching is literating over
* `p`: the patch center

# Parameters

* (Required) `num_patches`: the number of patches that should be returned.
* (Required) `patch_size`
* `search_size = size(img)`: the search window size, by default it searches the whole image.
* `stride = patch_size`: loop step size

# Examples

```jldoctest; setup = :(using TestImages, ImageNoise)
julia> img = testimage("cameraman");

julia> p = CartesianIndex(35, 35);

julia> indices = ReduceNoise.block_matching(img, p; num_patches=5, patch_size=7)
5-element Array{CartesianIndices{2,Tuple{UnitRange{Int64},UnitRange{Int64}}},1}:
[...]

julia> first(indices) == p # p is always the first item
true
```

"""
function block_matching(
        f, img, p::CartesianIndex;
        num_patches,
        patch_size,
        search_window_size=size(img),
        patch_stride=patch_size)::Vector{CartesianIndices{ndims(img)}}
    # Offset is not considered, it might work, it might not work. I don't know yet.
    Base.require_one_based_indexing(img)

    patch_size isa Number && (patch_size = ntuple(_ -> patch_size, ndims(img)))
    patch_stride isa Number && (patch_stride = ntuple(_ -> patch_stride, ndims(img)))
    search_window_size isa Number && (search_window_size = ntuple(_ -> search_window_size, ndims(img)))
    all(isodd.(patch_size)) || throw(ArgumentError("`patch_size = $(patch_size)` should be odd numbers"))

    R = CartesianIndices(img)
    rₚ = CartesianIndex(patch_size .÷ 2) # patch radius
    Base.@boundscheck checkbounds(R, p - rₚ)
    Base.@boundscheck checkbounds(R, p + rₚ)

    Δ = CartesianIndex(patch_stride)
    rₛ = CartesianIndex(search_window_size .÷ 2) # search window radius

    # for simplicity, only consider the patches that do not exceed the image boundary
    # This requires at least Julia 1.6: https://github.com/JuliaLang/julia/pull/37829
    I_first, I_last = first(R) + rₚ, last(R) - rₚ
    Rw = vec(max(I_first, p-rₛ):Δ:min(p+rₛ, I_last))
    length(Rw) <= num_patches && throw(ArgumentError("search window size $(search_window_size) is too small to get enough patches"))

    # patch_p is used repeatly so we need a contiguous memeory layout to get better performance
    # TODO: pre-allocate patch_p?
    patch_p = img[p - rₚ:p + rₚ]
    # patch_q = similar(patch_p)
    dist = map(Rw) do q
        patch_q = @view img[q - rₚ:q + rₚ]
        f(patch_p, patch_q)
    end
    dist_ranks = partialsortperm(dist, 1:num_patches)

    return [q - rₚ:q + rₚ for q in Rw[dist_ranks]]
end
block_matching(img, p::CartesianIndex; kwargs...) = block_matching(ssd, img, p::CartesianIndex; kwargs...)


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
