#include "utils.h"
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

template <typename scalar_t>
__global__ void haz_all_composite_train_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> p_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> p_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> total_samples,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> f_rgb,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> f_opacity,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_rgb,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_ws,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_opacity
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= c_sigmas.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    int samples = 0;
    scalar_t f_T = 1.0f;
    scalar_t c_T = 1.0f;


    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t one_a_c = __expf(-c_sigmas[s]*deltas[s]);
        const scalar_t one_a_p = __expf(-p_sigmas[s]*deltas[s]);


        f_rgb[ray_idx][0] += f_T*( (1.0f-one_a_c)*c_rgbs[s][0]+(1.0f-one_a_p)*p_rgbs[s][0]);
        f_rgb[ray_idx][1] += f_T*( (1.0f-one_a_c)*c_rgbs[s][1]+(1.0f-one_a_p)*p_rgbs[s][1]);
        f_rgb[ray_idx][2] += f_T*( (1.0f-one_a_c)*c_rgbs[s][2]+(1.0f-one_a_p)*p_rgbs[s][2]);
        f_opacity[ray_idx] += f_T*( (1.0f-one_a_c)+(1.0f-one_a_p) );
        f_T *= one_a_c*one_a_p;


        c_rgb[ray_idx][0] += c_T*(1.0f-one_a_c)*c_rgbs[s][0];
        c_rgb[ray_idx][1] += c_T*(1.0f-one_a_c)*c_rgbs[s][1];
        c_rgb[ray_idx][2] += c_T*(1.0f-one_a_c)*c_rgbs[s][2];
        c_ws[s] += c_T*(1.0f-one_a_c);
        c_opacity[ray_idx] += c_T*(1.0f-one_a_c);
        c_T *= one_a_c;


        if (f_T <= T_threshold) break; // ray has enough opacity
        samples++;
    }
    total_samples[n] = samples;
}


std::vector<torch::Tensor> haz_all_composite_train_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float T_threshold
){
    const int N_rays = rays_a.size(0), N = c_sigmas.size(0);

    auto f_rgb = torch::zeros({N_rays, 3}, c_sigmas.options());
    auto f_opacity = torch::zeros({N_rays}, c_sigmas.options());

    auto c_rgb = torch::zeros({N_rays, 3}, c_sigmas.options());
    auto c_ws = torch::zeros({N}, c_sigmas.options());
    auto c_opacity = torch::zeros({N_rays}, c_sigmas.options());

    auto total_samples = torch::zeros({N_rays}, torch::dtype(torch::kLong).device(c_sigmas.device()));
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_all_composite_train_fw_cu",
    ([&] {
        haz_all_composite_train_fw_kernel<scalar_t><<<blocks, threads>>>(
            c_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            p_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            p_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            T_threshold,
            total_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            f_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            f_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            c_ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {total_samples, f_rgb,f_opacity,c_rgb,c_ws,c_opacity};
}


template <typename scalar_t>
__global__ void haz_all_composite_train_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_df_rgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_df_opacity,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dc_rgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dc_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dc_opacity,
    scalar_t* __restrict__ dL_dc_ws_times_c_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> p_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> p_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> f_rgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> f_opacity,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_rgb,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_opacity,
    const scalar_t T_threshold,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dc_sigmas,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dp_sigmas,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dc_rgbs,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> dL_dp_rgbs
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= c_sigmas.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];


    int samples = 0;

    scalar_t f_R = f_rgb[ray_idx][0];
    scalar_t f_G = f_rgb[ray_idx][1];
    scalar_t f_B = f_rgb[ray_idx][2];
    scalar_t f_T = 1.0f;
    scalar_t f_T_i = 1.0f;
    scalar_t f_r = 0.0f;
    scalar_t f_g = 0.0f;
    scalar_t f_b = 0.0f;
    scalar_t f_O = f_opacity[ray_idx];
    scalar_t f_o = 0.0f;

    scalar_t c_T = 1.0f;
    scalar_t c_O = c_opacity[ray_idx];

    thrust::inclusive_scan(thrust::device,
                           dL_dc_ws_times_c_ws+start_idx,
                           dL_dc_ws_times_c_ws+start_idx+N_samples,
                           dL_dc_ws_times_c_ws+start_idx);
    scalar_t dL_dc_ws_times_c_ws_sum = dL_dc_ws_times_c_ws[start_idx+N_samples-1];

    while (samples < N_samples) {
        const int s = start_idx + samples;
        const scalar_t one_a_c = __expf(-c_sigmas[s]*deltas[s]);
        const scalar_t one_a_p = __expf(-p_sigmas[s]*deltas[s]);


        f_r += f_T*( (1.0f-one_a_c)*c_rgbs[s][0]+(1.0f-one_a_p)*p_rgbs[s][0]);
        f_g += f_T*( (1.0f-one_a_c)*c_rgbs[s][1]+(1.0f-one_a_p)*p_rgbs[s][1]);
        f_b += f_T*( (1.0f-one_a_c)*c_rgbs[s][2]+(1.0f-one_a_p)*p_rgbs[s][2]);
        f_o += f_T*(1.0f-one_a_p + 1.0f-one_a_c);
        f_T *= one_a_c*one_a_p;

        dL_dc_rgbs[s][0] = dL_df_rgb[ray_idx][0] * f_T_i * (1.0f-one_a_c) + dL_dc_rgb[ray_idx][0] * c_T * (1.0f - one_a_c);
        dL_dc_rgbs[s][1] = dL_df_rgb[ray_idx][1] * f_T_i * (1.0f-one_a_c) + dL_dc_rgb[ray_idx][0] * c_T * (1.0f - one_a_c);
        dL_dc_rgbs[s][2] = dL_df_rgb[ray_idx][2] * f_T_i * (1.0f-one_a_c) + dL_dc_rgb[ray_idx][0] * c_T * (1.0f - one_a_c);


        dL_dp_rgbs[s][0] = dL_df_rgb[ray_idx][0] * f_T_i * (1.0f-one_a_p) ;
        dL_dp_rgbs[s][1] = dL_df_rgb[ray_idx][1] * f_T_i * (1.0f-one_a_p) ;
        dL_dp_rgbs[s][2] = dL_df_rgb[ray_idx][2] * f_T_i * (1.0f-one_a_p) ;

        c_T *= one_a_c;

        dL_dc_sigmas[s] = deltas[s] * (
            dL_df_rgb[ray_idx][0]*(c_rgbs[s][0]*f_T_i*one_a_c-(f_R-f_r)) +
            dL_df_rgb[ray_idx][1]*(c_rgbs[s][1]*f_T_i*one_a_c-(f_G-f_g)) +
            dL_df_rgb[ray_idx][2]*(c_rgbs[s][2]*f_T_i*one_a_c-(f_B-f_b)) +
            dL_df_opacity[ray_idx]*( f_T_i*one_a_c-(f_O-f_o) ) +
            dL_dc_opacity[ray_idx]*(1-c_O) +
            c_T*dL_dc_ws[s] - ( dL_dc_ws_times_c_ws_sum - dL_dc_ws_times_c_ws[s] )
        );

        dL_dp_sigmas[s] = deltas[s] * (
            dL_df_rgb[ray_idx][0]*(p_rgbs[s][0]*f_T_i*one_a_p-(f_R-f_r)) +
            dL_df_rgb[ray_idx][1]*(p_rgbs[s][1]*f_T_i*one_a_p-(f_G-f_g)) +
            dL_df_rgb[ray_idx][2]*(p_rgbs[s][2]*f_T_i*one_a_p-(f_B-f_b)) +
            dL_df_opacity[ray_idx]*( f_T_i*one_a_p-(f_O-f_o) )
        );


        f_T_i *= one_a_c*one_a_p;
        if (f_T <= T_threshold) break;
        samples++;
    }
}


std::vector<torch::Tensor> haz_all_composite_train_bw_cu(
    const torch::Tensor dL_df_rgb,
    const torch::Tensor dL_df_opacity,
    const torch::Tensor dL_dc_rgb,
    const torch::Tensor dL_dc_ws,
    const torch::Tensor dL_dc_opacity,
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor f_rgb,
    const torch::Tensor f_opacity,
    const torch::Tensor c_rgb,
    const torch::Tensor c_ws,
    const torch::Tensor c_opacity,
    const float T_threshold
){
    const int N = c_sigmas.size(0), N_rays = rays_a.size(0);

    auto dL_dc_sigmas = torch::zeros({N}, c_sigmas.options());
    auto dL_dp_sigmas = torch::zeros({N}, c_sigmas.options());

    auto dL_dc_rgbs = torch::zeros({N, 3}, c_sigmas.options());
    auto dL_dp_rgbs = torch::zeros({N, 3}, c_sigmas.options());

    auto dL_dc_ws_times_c_ws = dL_dc_ws * c_ws; // auxiliary input

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_all_composite_train_bw_cu",
    ([&] {
        haz_all_composite_train_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_df_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_df_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dc_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dc_ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dc_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dc_ws_times_c_ws.data_ptr<scalar_t>(),
            c_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            p_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            p_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            f_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            f_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            c_ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            T_threshold,
            dL_dc_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dp_sigmas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            dL_dc_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            dL_dp_rgbs.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {dL_dc_sigmas,dL_dp_sigmas, dL_dc_rgbs, dL_dp_rgbs};
}






// --------------------------------------------------------------------------------------------------------------------------


template <typename scalar_t>
__global__ void all_composite_test_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> p_sigmas,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> c_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> p_rgbs,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> hits_t,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> alive_indices,
    const scalar_t T_threshold,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> N_eff_samples,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> c_rgb,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_depth,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_opacity,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> f_rgb,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> f_depth,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> f_opacity
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= alive_indices.size(0)) return;

    if(N_eff_samples[n] == 0){
        alive_indices[n] = -1;
        return;
    }

    const size_t r = alive_indices[n];

    int s = 0;
    scalar_t f_T = 1.0f-f_opacity[r];
    scalar_t c_T = 1.0f-c_opacity[r];

    while (s < N_eff_samples[n]) {

        const scalar_t one_a_c = __expf(-c_sigmas[n][s]*deltas[n][s]);
        const scalar_t one_a_p = __expf(-p_sigmas[n][s]*deltas[n][s]);


        c_rgb[r][0] += c_T*( (1.0f-one_a_c)*c_rgbs[n][s][0] );
        c_rgb[r][1] += c_T*( (1.0f-one_a_c)*c_rgbs[n][s][1] );
        c_rgb[r][2] += c_T*( (1.0f-one_a_c)*c_rgbs[n][s][2] );
        c_depth[r] += c_T* (1.0f-one_a_c) *ts[n][s];
        c_opacity[r] += c_T* (1.0f-one_a_c);
        c_T *= one_a_c;


        f_rgb[r][0] += f_T*( (1.0f-one_a_c)*c_rgbs[n][s][0]+(1.0f-one_a_p)*p_rgbs[n][s][0]);
        f_rgb[r][1] += f_T*( (1.0f-one_a_c)*c_rgbs[n][s][1]+(1.0f-one_a_p)*p_rgbs[n][s][1]);
        f_rgb[r][2] += f_T*( (1.0f-one_a_c)*c_rgbs[n][s][2]+(1.0f-one_a_p)*p_rgbs[n][s][2]);
        f_depth[r] += f_T*( (1.0f-one_a_c)+(1.0f-one_a_p) )*ts[n][s];
        f_opacity[r] += f_T*( (1.0f-one_a_c)+(1.0f-one_a_p) );
        f_T *= one_a_c*one_a_p;

        if (f_T <= T_threshold){
            alive_indices[n] = -1;
            break;
        }
        s++;
    }
}


void all_composite_test_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor c_rgb,
    torch::Tensor c_depth,
    torch::Tensor c_opacity,
    torch::Tensor f_rgb,
    torch::Tensor f_depth,
    torch::Tensor f_opacity
){
    const int N_rays = alive_indices.size(0);

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "all_composite_test_fw_cu",
    ([&] {
        all_composite_test_fw_kernel<scalar_t><<<blocks, threads>>>(
            c_sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            p_sigmas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            c_rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            p_rgbs.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            hits_t.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            alive_indices.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            T_threshold,
            N_eff_samples.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            c_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            c_depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            f_rgb.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            f_depth.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            f_opacity.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

}



