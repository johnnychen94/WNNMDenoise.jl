using WNNMDenoise
using MAT, FileIO
using ImageQualityIndexes

@assert VERSION >= v"1.7.0-beta4"

using LinearAlgebra
using MKL
@info BLAS.get_config() # MKL is faster on svd

for σₙ = [20, 40, 60, 80, 100]
    img = Float32.(load("house.png")) * 255;
    noisy_img = matopen(joinpath(@__DIR__, "data", "house_AWGN_$(σₙ).mat")) do io
        read(io, "N_Img")
    end

    start_psnr = assess_psnr(noisy_img, img, 255)

    out = copy(noisy_img)
    f = WNNM(σₙ)
    @time f(out, noisy_img, clean_img=img);

    end_psnr = assess_psnr(out, img, 255)
end
