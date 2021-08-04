module WNNMDenoise

using LinearAlgebra
using ImageCore
using ImageCore: NumberLike, GenericGrayImage, GenericImage
using ImageDistances
using Statistics
using BlockMatching

using ImageQualityIndexes # TODO: remove this
using ProgressMeter # TODO: remove this

include("utilities.jl")

export WNNM

"""
    WNNM(noise_level; kwargs...)

Weighted nuclear norm minimization denoising algorithm.

!!! note
    To keep consistent with the original implementation[2], here we stick
    to the 0-255 value range world. If the image is 0-1 range, then the default
    parameters may not work at all.

# Arguments

- `noise_level`: the (estimated) gaussian noise level, used in the first WNNM iteration.

# Keywords

The default values of these keywords come from the original reference
implementation [2]. Except that `patch_size` in our implementation has
to be odd number.

- `K`: the number of WNNM iterations
- `δ`: step value for each WNNM iteration

keywords that control the block matching step, it can be scalar or a vector of length `K`. If it is
a vector then it specifies the values for each WNNM iteration

- `num_patches`: the number of matched patches.
- `patch_size`: patch size.
- `patch_stride`: the stride of each sliding window.
- `window_size`: non-local search size in block matching.

keywords that control the internal WNNM solver, which is a small threshold-svd loop
on the extracted block matching results.

- `λ`: weight constant used to estimate the remained noise level of each patch.
- `C`: weight constant in WNNM solver used by the original implementation.

# Examples

```julia
using TestImages, WNNMDenoise, ImageTransformations

# 1. load a clean image
#    0-255 range is required
clean_img = Float64.(imresize(testimage("cameraman") .* 255, ratio=0.5));

# 2. add guassian noise
σₙ = 40;
noisy_img = clean_img .+ σₙ .* randn(Float32, size(clean_img))

# 3. denoise using the WNNM denoiser
f = WNNM(40)
denoised_img = f(copy(noisy_img), noisy_img)
```

# References

[1] Gu, Shuhang, et al. "Weighted nuclear norm minimization with application to image denoising." _Proceedings of the IEEE conference on computer vision and pattern recognition_. 2014.
[2] The MATLAB reference implementation: http://www4.comp.polyu.edu.hk/~cslzhang/code/WNNM_code.zip

"""
struct WNNM
    noise_level::Float64
    K::Int
    δ::Float64
    num_patches::Vector{Int}
    patch_size::Vector{Int}
    patch_stride::Vector{Int}
    λ::Float64
    C::Float64
    window_size::Int
end

function WNNM(noise_level;
              δ=0.1,
              C=2sqrt(2),
              window_size=60,
              patch_size=nothing,
              num_patches=nothing,
              K=nothing,
              λ=nothing,
              patch_stride=nothing)
    if noise_level <= 20
        isnothing(patch_size) && (patch_size = 5)
        isnothing(num_patches) && (num_patches = 70)
        isnothing(K) && (K = 8)
        isnothing(λ) && (λ = 0.56)
    elseif noise_level <= 40
        isnothing(patch_size) && (patch_size = 7)
        isnothing(num_patches) && (num_patches = 90)
        isnothing(K) && (K = 12)
        isnothing(λ) && (λ = 0.56)
    elseif noise_level <= 60
        isnothing(patch_size) && (patch_size = 7)
        isnothing(num_patches) && (num_patches = 120)
        isnothing(K) && (K = 14)
        isnothing(λ) && (λ = 0.58)
    else
        isnothing(patch_size) && (patch_size = 9)
        isnothing(num_patches) && (num_patches = 140)
        isnothing(K) && (K = 14)
        isnothing(λ) && (λ = 0.58)
    end

    @assert isodd(patch_size) "patch size is expected to be odd number, instead it is $patch_size"
    patch_size isa Number && (patch_size = fill(patch_size, K))
    isnothing(patch_stride) && (patch_stride = @. max(1, patch_size ÷ 2 - 1))
    patch_stride isa Number && (patch_stride = fill(patch_stride, K))

    num_patches = fill(num_patches - 10, K)
    drop_freq = 2
    for k in 2:K
        # drop by 10 for every 2 iteration
        num_patches[k] = (k - 1) % drop_freq == 0 ? num_patches[k - 1] - 10 : num_patches[k - 1]
    end

    WNNM(noise_level, K, δ, num_patches, patch_size, patch_stride, λ, C, window_size)
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
            )
        end

        # TODO: remove this logging part when it is ready
        if !isnothing(clean_img)
            @info "Result" iter psnr = assess_psnr(clean_img, imgₑₛₜ, 255) num_patches = f.num_patches[iter]
            display(Gray.(imgₑₛₜ ./ 255))
            sleep(0.1)
        end
    end
    return imgₑₛₜ
end

function _estimate_img(imgₑₛₜ, imgₙ; patch_size, patch_stride, num_patches, kwargs...)
    patch_size = ntuple(_ -> patch_size, ndims(imgₑₛₜ))

    # We set stride in both dimension instead only in column dimension.
    # This gives less computation but the similar output speaking of PSNR.
    # Δ = CartesianIndex(patch_stride, ntuple(_->1, ndims(imgₑₛₜ)-1)...) # original version in MATLAB
    Δ = CartesianIndex(ntuple(_->patch_stride, ndims(imgₑₛₜ)))

    r = CartesianIndex(patch_size .÷ 2)
    R = CartesianIndices(imgₑₛₜ)
    R = first(R) +r:Δ:last(R) -r

    imgₑₛₜ⁺ = zeros(eltype(imgₑₛₜ), axes(imgₑₛₜ))
    W = zeros(Int, axes(imgₑₛₜ))

    progress = Progress(length(R[1:patch_stride:end]))
    out_patches = [Matrix{eltype(imgₑₛₜ)}(undef, prod(patch_size), num_patches) for i in 1:Threads.nthreads()]
    patch_q_indices_buffer = [Matrix{CartesianIndex{2}}(undef, prod(patch_size), num_patches) for i in 1:Threads.nthreads()]
    Threads.@threads for p in R
        out = out_patches[Threads.threadid()]
        patch_q_indices = patch_q_indices_buffer[Threads.threadid()]

        fill!(out, zero(eltype(out)))
        patch_q_indices = _estimate_patch!(patch_q_indices, out, imgₑₛₜ, imgₙ, p; patch_size=patch_size, num_patches=num_patches, kwargs...)

        # Technically, there will be data racing here if multiple threads are involved, but as we've
        # observed, this doesn't affect the overall performance.
        view(W, patch_q_indices) .+= 1
        view(imgₑₛₜ⁺, patch_q_indices) .+= out
        next!(progress)
    end

    return imgₑₛₜ⁺ ./ max.(W, 1)
end

function _estimate_patch!(patch_q_indices, out, imgₑₛₜ, imgₙ, p;
                         noise_level,
                         patch_size::Tuple,
                         num_patches::Integer,
                         window_size,
                         λ,
                         C,
                         σₚ=nothing)
    rₚ = CartesianIndex(patch_size .÷ 2)
    p_indices = p - rₚ:p + rₚ

    alg = FullSearch(2;
        patch_radius=rₚ,
        search_radius=window_size.÷2,
        search_stride=1
    )
    q_inds = multi_match(alg, imgₑₛₜ, imgₑₛₜ, p; num_patches=num_patches)
    # the memory allocation will be a hospot if we directly generate `patch_q_indices` using `vcat`,
    # thus we pre-allocate it and then copyto the buffer.
    @inbounds for i = 1:length(q_inds)
        q = q_inds[i]
        copyto!(view(patch_q_indices, :, i), q - rₚ:q + rₚ)
    end
    m = mean(@view(imgₑₛₜ[patch_q_indices]); dims=2)

    if isnothing(σₚ)
        # Try: use the mean estimated σₚ of each patch
        σₚ = _estimate_noise_level(view(imgₑₛₜ, p_indices), view(imgₙ, p_indices), noise_level; λ=λ)
    end
    out .= @view(imgₑₛₜ[patch_q_indices]) .- m
    WNNM_optimizer!(out, out, eltype(out)(σₚ); C=C)
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
function WNNM_optimizer!(out, Y, σₚ; C, fixed_point_num_iters=3)
    # Apply Corollary 1 in [1] for image denoise purpose
    # Note: this solver is reserved to the denoising method and is not supposed to be used in other
    #       applications; it simply isn't designed so.

    n = size(Y, 2)
    U, ΣY, V = svd!(Y)

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
