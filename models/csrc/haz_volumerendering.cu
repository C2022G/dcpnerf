#include "utils.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>


template <typename scalar_t>
__global__ void haz_composite_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> a_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];


    int samples = 0;
    scalar_t T = 1.0f;
    scalar_t a_T = 1.0f;

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T; // weight of the sample point

        rgb[ray_idx][0] += w*rgbs[s][0];
        rgb[ray_idx][1] += w*rgbs[s][1];
        rgb[ray_idx][2] += w*rgbs[s][2];
        opacity[ray_idx] += w;
        ws[s] = w;
        T *= 1-a;
        a_T *= __expf(-(a_sigmas[s]+sigmas[s])*deltas[s]);

        if (a_T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[n] = samples;
}


std::vector<torch::Tensor> haz_composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor a_sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float T_threshold
){
    const int N_rays = rays_a.size(0), N = sigmas.size(0);

    auto opacity = torch::zeros({N_rays}, sigmas.options());
    auto rgb = torch::zeros({N_rays, 3}, sigmas.options());
    auto ws = torch::zeros({N}, sigmas.options());
    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(sigmas.device()));

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "haz_composite_train_fw_cu",
    ([&] {
        haz_composite_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            a_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, opacity, rgb, ws};
}


template <typename scalar_t>
__global__ void haz_composite_train_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dopacity,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws,
    scalar_t* __restrict__ dL_dws_times_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> a_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> opacity,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> rgb,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_drgbs
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= opacity.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // front to back compositing
    int samples = 0;
    scalar_t R = rgb[ray_idx][0], G = rgb[ray_idx][1], B = rgb[ray_idx][2];
    scalar_t O = opacity[ray_idx];
    scalar_t T = 1.0f, r = 0.0f, g = 0.0f, b = 0.0f,a_T = 1.0f;

    thrust::inclusive_scan(thrust::device,
                           dL_dws_times_ws+start_idx,
                           dL_dws_times_ws+start_idx+N_samples,
                           dL_dws_times_ws+start_idx);
    scalar_t dL_dws_times_ws_sum = dL_dws_times_ws[start_idx+N_samples-1];

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t a = 1.0f - __expf(-sigmas[s]*deltas[s]);
        const scalar_t w = a * T;

        r += w*rgbs[s][0]; g += w*rgbs[s][1]; b += w*rgbs[s][2];
        T *= 1-a;
        a_T *= __expf(-(a_sigmas[s]+sigmas[s])*deltas[s]);

        // compute gradients by math...
        dL_drgbs[s][0] = dL_drgb[ray_idx][0]*w;
        dL_drgbs[s][1] = dL_drgb[ray_idx][1]*w;
        dL_drgbs[s][2] = dL_drgb[ray_idx][2]*w;

        dL_dsigmas[s] = deltas[s] * (
            dL_drgb[ray_idx][0]*(rgbs[s][0]*T-(R-r)) +
            dL_drgb[ray_idx][1]*(rgbs[s][1]*T-(G-g)) +
            dL_drgb[ray_idx][2]*(rgbs[s][2]*T-(B-b)) + // gradients from rgb
            dL_dopacity[ray_idx]*(1-O) + // gradient from opacity
            T*dL_dws[s]-(dL_dws_times_ws_sum-dL_dws_times_ws[s]) // gradient from ws
        );



        if (a_T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
}


std::vector<torch::Tensor> haz_composite_train_bw_cu(
    const torch::Tensor dL_dopacity,
    const torch::Tensor dL_drgb,
    const torch::Tensor dL_dws,
    const torch::Tensor sigmas,
    const torch::Tensor a_sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor opacity,
    const torch::Tensor rgb,
    const float T_threshold
){
    const int N = sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dsigmas = torch::zeros({N}, sigmas.options());
    auto dL_drgbs = torch::zeros({N, 3}, sigmas.options());

    auto dL_dws_times_ws = dL_dws * ws; // auxiliary input

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigmas.type(), "haz_composite_train_bw_cu",
    ([&] {
        haz_composite_train_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dopacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dws_times_ws.data_ptr<scalar_t>(),
            sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            a_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            dL_dsigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_drgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dsigmas, dL_drgbs};
}

