module WNNMDenoise

using Base.Iterators
using LinearAlgebra
using OffsetArrays
using ImageCore
using ImageCore.MappedArrays
using ImageCore: NumberLike, GenericGrayImage, GenericImage
using ImageFiltering
using ImageDistances
using Statistics
using LowRankApprox

using ImageQualityIndexes # TODO: remove this
using ProgressMeter # TODO: remove this

include("utilities.jl")

export WNNM

"""

# References

[1] Gu, S., Zhang, L., Zuo, W., & Feng, X. (2014). Weighted nuclear norm minimization with application to image denoising. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2862-2869).

"""
struct WNNM
    "Estimated gaussian noise level"
    noise_level::Float64
    "Number of WNNM iterations"
    K::Int
    "step value for each WNNM iteration"
    δ::Float64
    "number of patches in each WNNM iteration"
    num_patches::Vector{Int}
    "patch size in each WNNM iteration"
    patch_size::Vector{Int}
    "patch stride in each WNNM iteration"
    patch_stride::Vector{Int}
    "weight constant used to estimate the remained noise level of each patch"
    λ::Float64
    "weight constant in WNNM solver"
    C::Float64
    "non-local search size in block matching"
    window_size::Int
    "matrix rank that used to allow early-stop in svd approximation"
    svd_rank::Vector{Int}
end

function WNNM(noise_level;
              δ=0.1,
              C=2sqrt(2),
              window_size=60,
              patch_size=nothing,
              num_patches=nothing,
              K=nothing,
              λ=nothing,
              svd_rank=nothing,
              patch_stride=nothing)
    if noise_level <= 20
        isnothing(patch_size) && (patch_size = 6)
        isnothing(num_patches) && (num_patches = 70)
        isnothing(K) && (K = 8)
        isnothing(λ) && (λ = 0.56)
    elseif noise_level <= 40
        isnothing(patch_size) && (patch_size = 7)
        isnothing(num_patches) && (num_patches = 90)
        isnothing(K) && (K = 12)
        isnothing(λ) && (λ = 0.56)
    elseif noise_level <= 60
        isnothing(patch_size) && (patch_size = 8)
        isnothing(num_patches) && (num_patches = 120)
        isnothing(K) && (K = 14)
        isnothing(λ) && (λ = 0.58)
    else
        isnothing(patch_size) && (patch_size = 9)
        isnothing(num_patches) && (num_patches = 140)
        isnothing(K) && (K = 14)
        isnothing(λ) && (λ = 0.58)
    end

    patch_size isa Number && (patch_size = fill(patch_size, K))
    isnothing(patch_stride) && (patch_stride = @. max(1, patch_size ÷ 2 - 1))
    patch_stride isa Number && (patch_stride = fill(patch_stride, K))

    if isnothing(svd_rank)
        svd_rank = map(enumerate(patch_size)) do (i, sz)
            min(10 + 5i, 10 + sz^2 ÷ 2)
        end
    elseif svd_rank isa Number
        svd_rank = fill(svd_rank, K)
    end

    num_patches = fill(num_patches - 10, K)
    drop_freq = 2
    for k in 2:K
        # drop by 10 for every 2 iteration
        num_patches[k] = (k - 1) % drop_freq == 0 ? num_patches[k - 1] - 10 : num_patches[k - 1]
    end

    WNNM(noise_level, K, δ, num_patches, patch_size, patch_stride, λ, C, window_size, svd_rank)
end

## Implementation

function (f::WNNM)(imgₑₛₜ, imgₙ; clean_img=nothing)
    if imgₑₛₜ === imgₙ
        imgₑₛₜ = copy(imgₙ)
    else
        copyto!(imgₑₛₜ, imgₙ)
    end

    T = eltype(eltype(imgₑₛₜ))
    for iter in 1:f.K
        @. imgₑₛₜ = imgₑₛₜ + f.δ * (imgₙ - imgₑₛₜ) # This iteration can be done more sophisticatedly

        # The noise level for the first iteration is known (whether it is estimated outside or a
        # white noise). The noise is removed in each iteration, so we have to estimate a noise level
        # at a patch level; the denoising performance on each patch can be different, which means a
        # global noise level can be misleading.
        σₚ = iter == 1 ? f.noise_level : nothing

        # calculating svd using blas threads is not an optimal parallel strategy
        imgₑₛₜ .= with_blas_threads(1) do
            _estimate_img(imgₑₛₜ, imgₙ;
                noise_level=f.noise_level,
                patch_size=f.patch_size[iter],
                patch_stride=f.patch_stride[iter],
                num_patches=f.num_patches[iter],
                window_size=f.window_size,
                λ=T(f.λ),
                C=T(f.C),
                σₚ=σₚ,
                svd_rank=f.svd_rank[iter],
            )
        end

        # TODO: remove this logging part when it is ready
        if !isnothing(clean_img)
            @info "Result" iter psnr = assess_psnr(clean_img, imgₑₛₜ, 255) num_patches = f.num_patches[iter] svd_rank=f.svd_rank[iter]
            display(Gray.(imgₑₛₜ ./ 255))
            sleep(0.1)
        end
    end
    return imgₑₛₜ
end

function _estimate_img(imgₑₛₜ, imgₙ; patch_size, patch_stride, num_patches, kwargs...)
    patch_size = ntuple(_ -> patch_size, ndims(imgₑₛₜ))

    # only set key patch stride in one dimension, this gives better denoising performance (Why?)
    # This also follow the original implementation
    Δ = CartesianIndex(patch_stride, ntuple(_->1, ndims(imgₑₛₜ)-1)...)
    # Δ = CartesianIndex(ntuple(_->patch_stride, ndims(imgₑₛₜ)))
    
    r = CartesianIndex(patch_size .÷ 2)
    R = CartesianIndices(imgₑₛₜ)
    R = first(R) +r:Δ:last(R) -r

    imgₑₛₜ⁺ = zeros(eltype(imgₑₛₜ), axes(imgₑₛₜ))
    W = zeros(Int, axes(imgₑₛₜ))

    progress = Progress(length(R[1:patch_stride:end]))
    out_patches = [Matrix{eltype(imgₑₛₜ)}(undef, prod(patch_size), num_patches) for i in 1:Threads.nthreads()]
    Threads.@threads for p in R
        out = out_patches[Threads.threadid()]
        fill!(out, zero(eltype(out)))
        patch_q_indices = _estimate_patch!(out, imgₑₛₜ, imgₙ, p; patch_size=patch_size, num_patches=num_patches, kwargs...)

        view(W, patch_q_indices) .+= 1
        view(imgₑₛₜ⁺, patch_q_indices) .+= out
        next!(progress)
    end

    return imgₑₛₜ⁺ ./ max.(W, 1)
end

function _estimate_patch!(out, imgₑₛₜ, imgₙ, p;
                         noise_level,
                         patch_size::Tuple,
                         num_patches::Integer,
                         window_size,
                         λ,
                         C,
                         svd_rank,
                         σₚ=nothing)
    rₚ = CartesianIndex(patch_size .÷ 2)
    p_indices = p - rₚ:p + rₚ

    patch_q_indices = block_matching(imgₑₛₜ, p;
        num_patches=num_patches,
        patch_size=patch_size,
        search_window_size=window_size,
        patch_stride=1
    )
    patch_q_indices = hcat([indices[:] for indices in patch_q_indices]...)
    m = mean(@view(imgₑₛₜ[patch_q_indices]); dims=2)

    if isnothing(σₚ)
        # Try: use the mean estimated σₚ of each patch
        σₚ = _estimate_noise_level(view(imgₑₛₜ, p_indices), view(imgₙ, p_indices), noise_level; λ=λ)
    end
    out .= @view(imgₑₛₜ[patch_q_indices]) .- m
    WNNM_optimizer!(out, out, eltype(out)(σₚ); C=C, svd_rank=svd_rank)
    out .+= m

    return patch_q_indices
end


@doc raw"""
    WNNM_optimizer(Y, σₚ; C, rank, fixed_point_num_iters=3)

Optimizes the weighted nuclear norm minimization problem with a fixed point estimation

```math
    \min_X \lVert Y - X \rVert^2_{F} + \lVert X \rVert_{w, *}
```

The weight `w` is specially chosen so that it satisfies the condition of Corollary 1 in [1].

# References

[1] Gu, S., Zhang, L., Zuo, W., & Feng, X. (2014). Weighted nuclear norm minimization with application to image denoising. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2862-2869).

"""
function WNNM_optimizer!(out, Y, σₚ; C, svd_rank, fixed_point_num_iters=3)
    # Apply Corollary 1 in [1] for image denoise purpose
    # Note: this solver is reserved to the denoising method and is not supposed to be used in other
    #       applications; it simply isn't designed so.

    # This is different from the original implementation. Here we use an approximate version of svd;
    # it gives better performance in both speed and denoising result.
    n = size(Y, 2)
    if svd_rank >= ceil(Int, 0.5minimum(size(Y)))
        U, ΣY, V = svd!(Y)
    else
        U, ΣY, V = LowRankApprox.psvd(Y; rank=svd_rank)
    end

    # For image denoising problems, it is natural to shrink large singular value less, i.e., to set
    # smaller weight to large singular value. For this reason, it uses `w = (C * sqrt(n))/(ΣX + eps())`
    # as the weights; inversely propotional to ΣX. With singular values ΣX sorted ascendingly, the
    # condition for Corollary 1 holds, and thus we could directly get the desired solution in a single
    # step.

    # Here we iterate more than once because we don't know what ΣX is; we have to iterate it a while 
    # from ΣY to get an relatively good estimation of it.
    # TODO: could we set default σₚ as 0?
    # TODO: is this the best initialization we can get?
    ΣX = @. sqrt(max(ΣY^2 - n * σₚ^2, 0))
    for _ in 1:fixed_point_num_iters
        # the iterative algorithm proposed in section 2.2.2 in [1]
        # Step 1 in the iterative algorithm becomes trivial and a no-op

        # Step 2 degenerates to a soft thresholding; both P and Q are identity matrix.
        # all in one line to avoid unnecessary allocation for temporarily variable w
        @. ΣX = soft_threshold(ΣY, (C * sqrt(n) * σₚ^2) / (ΣX + eps()))
    end

    mul!(out, rmul!(U, Diagonal(ΣX)), V')
end


function _estimate_noise_level(patchₑₛₜ, patchₙ, σₙ; λ=0.56)
    # Estimate the noise level of given patch during the WNNM iteration
    # we still need to know the input noisy level σₙ to give an estimation
    λ * sqrt(abs(σₙ^2 - mse(patchₑₛₜ, patchₙ)))
end


end
