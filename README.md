# WNNMDenoise

![](https://img.shields.io/badge/Julia-1.6-blue)
[![Build Status](https://github.com/johnnychen94/WNNMDenoise.jl/workflows/CI/badge.svg)](https://github.com/johnnychen94/WNNMDenoise.jl/actions)
[![Coverage](https://codecov.io/gh/johnnychen94/WNNMDenoise.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/johnnychen94/WNNMDenoise.jl)

The julia implementation of WNNM denoising algorithm. This repo is only for archive and benchmark
purpose.

## Noteworthy difference from the original implementation


Performance tricks:

- The block matching stride is set in both dimension, while in the original implementation [2] this
  is only set in one dimension, with the other dimension stride being 1. This extra computation
  brings almost no benefit speaking of the PSNR/runtime. For example, when noise level is `40` and
  with the same default parameters, the overall PSNR is `31.30` in about 50 seconds, if we set
  stride in both dimension, then PNSR is `31.29` in about 26 seconds.
- When doing block matching, we sample the patch into a smaller one by setting indexing stride `2`.


## References

[1] Gu, S., Zhang, L., Zuo, W., & Feng, X. (2014). Weighted nuclear norm minimization with application to image denoising. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 2862-2869).

[2] The MATLAB reference implementation: http://www4.comp.polyu.edu.hk/~cslzhang/code/WNNM_code.zip
