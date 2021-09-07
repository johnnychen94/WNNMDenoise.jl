using Images, WNNMDenoise, TestImages
using Random
using MKL

img = imresize(Float64.(load("house.png")) * 255, ratio=0.5);
img = Float32.(load("house.png")) * 255;

clean_img = img;
σₙ = 40;
Random.seed!(MersenneTwister(), 0)
noisy_img = clean_img .+ σₙ .* randn(Float32, size(clean_img))

assess_psnr(noisy_img, img, 255)

out = copy(noisy_img)
f = WNNM(σₙ)
@time f(out, noisy_img; clean_img=clean_img)
