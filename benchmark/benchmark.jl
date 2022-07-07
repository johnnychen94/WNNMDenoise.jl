# Performance and runtime benchmark to the original MATLAB implementation

# julia> versioninfo()
# Julia Version 1.7.0-beta4
# Commit d0c90f37ba (2021-08-24 12:35 UTC)
# Platform Info:
#   OS: macOS (x86_64-apple-darwin19.6.0)
#   CPU: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
#   WORD_SIZE: 64
#   LIBM: libopenlibm
#   LLVM: libLLVM-12.0.1 (ORCJIT, skylake)
# Environment:
#   JULIA_NUM_THREADS = 8

using WNNMDenoise
using MAT, FileIO
using ImageQualityIndexes

using LinearAlgebra
using MKL
@info BLAS.get_config() # MKL is faster on svd

img = Float32.(load("house.png")) * 255;

σₙ = 40
noisy_img = matopen(joinpath("benchmark", "data", "house_AWGN_$(σₙ).mat")) do io
    read(io, "N_Img")
end

assess_psnr(noisy_img, img, 255)
# MATLAB: 16.06
# Julia: 16.06

out = copy(noisy_img)
# f = WNNM(σₙ, K=2)
f = WNNM(σₙ)
# @time f(out, noisy_img; clean_img=img);
@time f(out, noisy_img);
# MATLAB: 97.365962 seconds
# Julia:  17.232886 seconds

assess_psnr(out, img, 255)
# MATLAB: 31.31
# Julia:  31.35
