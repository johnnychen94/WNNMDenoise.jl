using WNNMDenoise
using ImageTransformations, TestImages, FileIO, ImageQualityIndexes
using Random
using Test

@testset "WNNMDenoise.jl" begin
    imgfile = joinpath(pkgdir(WNNMDenoise), "house.png")
    img = imresize(Float64.(load(imgfile)) * 255, ratio=0.5);
    clean_img = img;
    σₙ = 40;
    noisy_img = clean_img .+ σₙ .* randn(MersenneTwister(0), Float32, size(clean_img))

    @test assess_psnr(noisy_img, img, 255) ≈ 16.08294 atol=1e-5

    out = copy(noisy_img)
    f = WNNM(σₙ)
    f(out, noisy_img)

    println("PSNR: ", assess_psnr(out, img, 255))
    @test assess_psnr(out, img, 255) >= 28.8
end
