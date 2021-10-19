using Plots, StatsPlots
using DataFrames, FileIO, CSVFiles
# gr()
pyplot()

palette = [:red, :green, :blue, :cyan, :black]
noise_levels = [20, 40, 60, 80, 100]

root = isdir("results") ? "." : "benchmark"
result_dir = joinpath(root, "results")
experiments = filter!(readdir(result_dir, join=true)) do expdir
    isdir(expdir) || return false
    any(endswith(".csv"), readdir(expdir))
end

default(
    titlefont = (15, "times"),
    legendfontsize = 10,
    size=(1000, 700),
    guidefont = (15, :black),
    dpi=300,
    # tickfont = (10, :orange),
    # guide = "x",
    # framestyle = :zerolines,
    yminorgrid = true
)

for expdir in experiments
    expname = basename(expdir)
    infofile = joinpath(expdir, "versioninfo.txt")
    @assert isfile(infofile) "versioninfo.txt does not exist"
    m = match(r"\d+", filter!(contains("JULIA_NUM_THREADS"), readlines(infofile))[1])
    @assert !isnothing(m) "versioninfo.txt file must contain JULIA_NUM_THREADS info"
    num_threads = m.match

    julia_results = map(noise_levels) do σ
        julia_file = joinpath(expdir, "julia_$σ.csv")
        df = DataFrame(load(julia_file))
        accumulate!(+, df[!, :runtime], df[!, :runtime])
        df
    end
    matlab_results = map(noise_levels) do σ
        matlab_file = joinpath(expdir, "matlab_$σ.csv")
        df = DataFrame(load(matlab_file))
        accumulate!(+, df[!, :runtime], df[!, :runtime])
        df
    end

    p1 = plot(
        title="$expname (n=$num_threads)",
        ylabel="Time (seconds) in log scale",
        xlabel="PSNR(dB)",
        legend=:bottomright,
        xlim=(10, 50),
    )
    for i in 1:length(noise_levels)
        julia_df = julia_results[i]
        matlab_df = matlab_results[i]
        c = palette[i]
        σ = noise_levels[i]

        plot!(p1, matlab_df[!, :psnr], matlab_df[!, :runtime], marker=:circle, label="MATLAB(σ=$σ)", yscale=:log10, c=c, line=:dash)
        plot!(p1, julia_df[!, :psnr], julia_df[!, :runtime], marker=:square, label="Julia(σ=$σ)", yscale=:log10, c=c, line=:solid)
    end
    p1

    julia_runtime = map(julia_results) do df
        last(df[!, :runtime])
    end
    matlab_runtime = map(matlab_results) do df
        last(df[!, :runtime])
    end

    p2 = plot(
        title="Runtime",
        xlabel="Noise levels",
        ylabel="Runtime (seconds)",
        legend=:topleft
    )
    ctg = repeat(["Julia", "MATLAB"], inner=5)
    nam = repeat(noise_levels, outer = 2)
    groupedbar!(p2, nam, [julia_runtime matlab_runtime]; group=ctg)

    p3 = plot(
        title="Runtime (Scaled)",
        ylabel="runtime ratio",
        xlabel="Noise levels",
        legend=:topleft
    )
    groupedbar!(p3, nam, [ones(size(matlab_runtime)) matlab_runtime./julia_runtime]; group=ctg)

    p = plot(p1, plot(p2, p3, layout=(2, 1)), layout=(1, 2))
    outfile = joinpath(result_dir, basename(expdir)*".png")
    savefig(p, outfile)
end
