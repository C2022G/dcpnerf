#include "utils.h"
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>




// for details of the formulae, please see https://arxiv.org/pdf/2206.05085.pdf

template <typename scalar_t>
__global__ void prefix_sums_kernel(
    const scalar_t* __restrict__ ws,
    const scalar_t* __restrict__ wts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ ws_inclusive_scan,
    scalar_t* __restrict__ ws_exclusive_scan,
    scalar_t* __restrict__ wts_inclusive_scan,
    scalar_t* __restrict__ wts_exclusive_scan
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // compute prefix sum of ws and ws*ts
    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_inclusive_scan+start_idx);
    thrust::inclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_inclusive_scan+start_idx);
    // [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
    thrust::exclusive_scan(thrust::device,
                           ws+start_idx,
                           ws+start_idx+N_samples,
                           ws_exclusive_scan+start_idx);
    thrust::exclusive_scan(thrust::device,
                           wts+start_idx,
                           wts+start_idx+N_samples,
                           wts_exclusive_scan+start_idx);
}




// --------------------------------------------------------------------------------------------------------------------------

template <typename scalar_t>
__global__ void distortion_loss_fw_kernel(
    const scalar_t* __restrict__ _loss,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    loss[ray_idx] = thrust::reduce(thrust::device,
                                   _loss+start_idx,
                                   _loss+start_idx+N_samples,
                                   (scalar_t)0);
}


std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto wts = ws * ts;

    auto ws_inclusive_scan = torch::zeros({N}, ws.options());
    auto ws_exclusive_scan = torch::zeros({N}, ws.options());
    auto wts_inclusive_scan = torch::zeros({N}, ws.options());
    auto wts_exclusive_scan = torch::zeros({N}, ws.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu_prefix_sums", 
    ([&] {
        prefix_sums_kernel<scalar_t><<<blocks, threads>>>(
            ws.data_ptr<scalar_t>(),
            wts.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            ws_inclusive_scan.data_ptr<scalar_t>(),
            ws_exclusive_scan.data_ptr<scalar_t>(),
            wts_inclusive_scan.data_ptr<scalar_t>(),
            wts_exclusive_scan.data_ptr<scalar_t>()
        );
    }));

    auto _loss = 2*(wts_inclusive_scan*ws_exclusive_scan-ws_inclusive_scan*wts_exclusive_scan) + 1.0f/3*ws*ws*deltas;

    auto loss = torch::zeros({N_rays}, ws.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_fw_cu", 
    ([&] {
        distortion_loss_fw_kernel<scalar_t><<<blocks, threads>>>(
            _loss.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return {loss, ws_inclusive_scan, wts_inclusive_scan};
}


template <typename scalar_t>
__global__ void distortion_loss_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> wts_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ws,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dws
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = rays_a[n][2];
    const int end_idx = start_idx+N_samples-1;

    const scalar_t ws_sum = ws_inclusive_scan[end_idx];
    const scalar_t wts_sum = wts_inclusive_scan[end_idx];
    // fill in dL_dws from start_idx to end_idx
    for (int s=start_idx; s<=end_idx; s++){
        dL_dws[s] = dL_dloss[ray_idx] * 2 * (
            (s==start_idx?
                (scalar_t)0:
                (ts[s]*ws_inclusive_scan[s-1]-wts_inclusive_scan[s-1])
            ) + 
            (wts_sum-wts_inclusive_scan[s]-ts[s]*(ws_sum-ws_inclusive_scan[s]))
        );
        dL_dws[s] += dL_dloss[ray_idx] * (scalar_t)2/3*ws[s]*deltas[s];
    }
}


torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    const int N_rays = rays_a.size(0), N = ws.size(0);

    auto dL_dws = torch::zeros({N}, dL_dloss.options());

    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(ws.type(), "distortion_loss_bw_cu", 
    ([&] {
        distortion_loss_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            wts_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            dL_dws.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dws;
}



// --------------------------------------------------------------------------------------------------------------------------



template <typename scalar_t>
__global__ void exclusive_sums_kernel(
    const scalar_t* __restrict__ w,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ w_exclusive_scan
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // [a0, a1, a2, a3, ...] -> [0, a0, a0+a1, a0+a1+a2, ...]
    thrust::exclusive_scan(thrust::device,
                           w+start_idx,
                           w+start_idx+N_samples,
                           w_exclusive_scan+start_idx);
}

template <typename scalar_t>
__global__ void inclusive_sums_kernel(
    const scalar_t* __restrict__ w,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    scalar_t* __restrict__ w_inclusive_scan
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int start_idx = rays_a[n][1], N_samples = rays_a[n][2];

    // [a0, a1, a2, a3, ...] -> [a0, a0+a1, a0+a1+a2, a0+a1+a2+a3, ...]
    thrust::inclusive_scan(thrust::device,
                           w+start_idx,
                           w+start_idx+N_samples,
                           w_inclusive_scan+start_idx);
}


template <typename scalar_t>
__global__ void alpha_rats_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> alpha,
    const scalar_t* __restrict__ alpha_ite,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> alpha_rats
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = vr_samples[n];

    const scalar_t alpha_sum = thrust::reduce(thrust::device,
                                   alpha_ite+start_idx,
                                   alpha_ite+start_idx+N_samples,
                                   (scalar_t)0);
    if(alpha_sum==0) return;
    const scalar_t alpha_sum_inv = 1.0f / alpha_sum;
    for(int i=0;i<N_samples;i++){
        const int s = start_idx+i;
        alpha_rats[s] = alpha[s] * alpha_sum_inv;
    }
}
// --------------------------------------------------------------------------------------------------------------------------

torch::Tensor alpha_rat_fw_cu(
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = sigma.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    auto alpha = 1.0f - torch::exp(-sigma*deltas);
    auto alpha_rat = torch::zeros({N},sigma.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigma.type(), "alpha_rat_fw_cu",
    ([&] {
        alpha_rats_kernel<scalar_t><<<blocks, threads>>>(
            alpha.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            alpha.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            alpha_rat.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));
    return alpha_rat;
}


template <typename scalar_t>
__global__ void alpha_rat_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const scalar_t* __restrict__ alpha_rat_dloss_ite,
    const scalar_t* __restrict__ alpha_ite,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> sigma,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> deltas,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigma
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = vr_samples[n];

    const scalar_t alpha_rat_dloss_sum = thrust::reduce(thrust::device,
                                   alpha_rat_dloss_ite+start_idx,
                                   alpha_rat_dloss_ite+start_idx+N_samples,
                                   (scalar_t)0);
    const scalar_t alpha_sum = thrust::reduce(thrust::device,
                                   alpha_ite+start_idx,
                                   alpha_ite+start_idx+N_samples,
                                   (scalar_t)0);

    for(int i=0;i<N_samples;i++){
        const scalar_t s = start_idx+i;
        dL_dsigma[s] = ( dL_dloss[s] - alpha_rat_dloss_sum ) / alpha_sum  *  __expf(-sigma[s]*deltas[s]) * deltas[s];
    }
}


torch::Tensor alpha_rat_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor alpha_rat,
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = sigma.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    auto alpha = 1.0f - torch::exp(-sigma*deltas);
    auto alpha_rat_dloss = dL_dloss * alpha_rat;

    auto dL_dsigma = torch::zeros({N}, dL_dloss.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sigma.type(), "alpha_rat_bw_cu",
    ([&] {
        alpha_rat_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            alpha_rat_dloss.data_ptr<scalar_t>(),
            alpha.data_ptr<scalar_t>(),
            sigma.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            deltas.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            dL_dsigma.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dsigma;
}



// --------------------------------------------------------------------------------------------------------------------------

template <typename scalar_t>
__global__ void haz_clear_weight_loss_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> c_alpha_rat,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> I_1_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> I_2_inclusive_scan,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> ts,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> weights
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = vr_samples[n];

    const scalar_t I_1_sum = I_1_inclusive_scan[start_idx+N_samples-1];
    const scalar_t I_2_sum = I_2_inclusive_scan[start_idx+N_samples-1];

    for(int i=0;i<N_samples;i++){
        const scalar_t s = start_idx+i;
        weights[s] = c_alpha_rat[s] * (  I_1_sum - 2*I_1_inclusive_scan[s] + (1.0f - c_alpha_rat[s])*ts[s] - (  I_2_sum - 2*I_2_inclusive_scan[s] + (1.0f - c_alpha_rat[s]) ) * ts[s]  );
    }
}


torch::Tensor haz_clear_weight_loss_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = c_sigmas.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    auto c_alpha = 1.0f - torch::exp(-c_sigmas*deltas);
    auto c_alpha_rat = torch::zeros({N},c_sigmas.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_clear_weight_sums_alhpa_rat",
    ([&] {
        alpha_rats_kernel<scalar_t><<<blocks, threads>>>(
            c_alpha.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            c_alpha.data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            c_alpha_rat.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    auto I_1_inclusive_scan = torch::zeros({N},c_sigmas.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_clear_weight_inclusive_prefix_sums",
    ([&] {
        inclusive_sums_kernel<scalar_t><<<blocks, threads>>>(
            ((1.0f - c_alpha_rat)*ts).data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            I_1_inclusive_scan.data_ptr<scalar_t>()
        );
    }));


    auto I_2_inclusive_scan = torch::zeros({N},c_sigmas.options());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_clear_weight_2_inclusive_prefix_sums",
    ([&] {
        inclusive_sums_kernel<scalar_t><<<blocks, threads>>>(
            (1.0f - c_alpha_rat).data_ptr<scalar_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            I_2_inclusive_scan.data_ptr<scalar_t>()
        );
    }));


    auto weights = torch::zeros({N}, c_sigmas.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(c_sigmas.type(), "haz_clear_weight_loss_fw_cu",
    ([&] {
        haz_clear_weight_loss_fw_kernel<scalar_t><<<blocks, threads>>>(
            c_alpha_rat.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            I_1_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            I_2_inclusive_scan.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            ts.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            weights.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return weights;
}

// --------------------------------------------------------------------------------------------------------------------------


template <typename scalar_t>
__global__ void foggy_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> weights,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> alpha_rat,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> loss
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = vr_samples[n];

    for(int i=0;i<N_samples;i++){
        const int s = start_idx + i;
        loss[ray_idx] +=  weights[s] * ( alpha_rat[s]*__logf(alpha_rat[s]) + 0.37 );
    }

}


torch::Tensor foggy_fw_cu(
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = alpha_rat.size(0);
    const int threads = 256, blocks = (N_rays+threads-1)/threads;

    auto loss = torch::zeros({N_rays}, alpha_rat.options());


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(alpha_rat.type(), "foggy_fw_cu",
    ([&] {
        foggy_fw_kernel<scalar_t><<<blocks, threads>>>(
            weights.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            alpha_rat.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            loss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return loss;
}


template <typename scalar_t>
__global__ void foggy_bw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dloss,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> weights,
    const torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> alpha_rat,
    const torch::PackedTensorAccessor64<int64_t, 2, torch::RestrictPtrTraits> rays_a,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> vr_samples,
    torch::PackedTensorAccessor<scalar_t, 1, torch::RestrictPtrTraits, size_t> dL_dsigma
){
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= rays_a.size(0)) return;

    const int ray_idx = rays_a[n][0], start_idx = rays_a[n][1], N_samples = vr_samples[n];


    for(int i=0;i<N_samples;i++){
        const int s = start_idx+i;
        dL_dsigma[s] = dL_dloss[ray_idx] * weights[s] * ( __logf(alpha_rat[s]) + 1 );
    }

}


torch::Tensor foggy_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    const int N_rays = rays_a.size(0), N = alpha_rat.size(0);

    auto dL_dsigma = torch::zeros({N}, dL_dloss.options());
    const int threads = 256, blocks = (N_rays+threads-1)/threads;



    AT_DISPATCH_FLOATING_TYPES_AND_HALF(alpha_rat.type(), "foggy_bw_cu",
    ([&] {
        foggy_bw_kernel<scalar_t><<<blocks, threads>>>(
            dL_dloss.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            weights.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            alpha_rat.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>(),
            rays_a.packed_accessor64<int64_t, 2, torch::RestrictPtrTraits>(),
            vr_samples.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            dL_dsigma.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits, size_t>()
        );
    }));

    return dL_dsigma;
}


// --------------------------------------------------------------------------------------------------------------------------
