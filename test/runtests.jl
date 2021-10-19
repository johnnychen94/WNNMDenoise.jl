using WNNMDenoise
using ImageTransformations, FileIO, ImageQualityIndexes, MAT
using Test

@testset "WNNMDenoise.jl" begin
    imgfile = joinpath(pkgdir(WNNMDenoise), "house.png")
    img = Float64.(load(imgfile) * 255)

    σₙ = 40
    noisy_img_file = joinpath(pkgdir(WNNMDenoise), "test", "house_AWGN_$(σₙ).mat")
    noisy_img = matopen(noisy_img_file) do io
        read(io, "N_Img")
    end

    @test assess_psnr(noisy_img, img, 255) ≈ 16.06 atol=1e-2

    img, noisy_img = restrict.((img, noisy_img))
    out = copy(noisy_img)
    f = WNNM(σₙ)
    f(out, noisy_img)

    println("PSNR: ", assess_psnr(out, img, 255))
    @test assess_psnr(out, img, 255) >= 28.18
end
