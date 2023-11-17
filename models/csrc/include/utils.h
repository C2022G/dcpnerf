#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> ray_aabb_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
);


std::vector<torch::Tensor> ray_sphere_intersect_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
);


void packbits_cu(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
);


torch::Tensor morton3D_cu(const torch::Tensor coords);
torch::Tensor morton3D_invert_cu(const torch::Tensor indices);


std::vector<torch::Tensor> raymarching_train_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor hits_t,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const torch::Tensor noise,
    const int grid_size,
    const int max_samples
);


std::vector<torch::Tensor> raymarching_test_cu(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const torch::Tensor density_bitfield,
    const int cascades,
    const float scale,
    const float exp_step_factor,
    const int grid_size,
    const int max_samples,
    const int N_samples
);


std::vector<torch::Tensor> distortion_loss_fw_cu(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
);


torch::Tensor distortion_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
);





std::vector<torch::Tensor> haz_composite_train_fw_cu(
    const torch::Tensor sigmas,
    const torch::Tensor a_sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
);


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
    const float opacity_threshold
);




std::vector<torch::Tensor> all_composite_train_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
);


std::vector<torch::Tensor> all_composite_train_bw_cu(
    const torch::Tensor dL_df_rgb,
    const torch::Tensor dL_df_opacity,
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor f_rgb,
    const torch::Tensor f_opacity,
    const float T_threshold
);



void all_composite_test_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor c_rgb,
    torch::Tensor c_depth,
    torch::Tensor c_opacity,
    torch::Tensor f_rgb,
    torch::Tensor f_depth,
    torch::Tensor f_opacity
);


void haz_composite_test_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor hits_t,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor f_rgb,
    torch::Tensor f_depth,
    torch::Tensor f_opacity
);



void composite_test_fw_cu(
    const torch::Tensor f_sigmas,
    const torch::Tensor sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor alive_indices,
    const float T_threshold,
    const torch::Tensor N_eff_samples,
    torch::Tensor opacity,
    torch::Tensor f_opacity,
    torch::Tensor depth,
    torch::Tensor rgb
);







std::vector<torch::Tensor> haz_depath_loss_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);


torch::Tensor haz_depath_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor alpha_rats,
    const torch::Tensor weights,
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples,
    const torch::Tensor a_sum
);




torch::Tensor foggy_fw_cu(
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);


torch::Tensor foggy_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);



torch::Tensor alpha_rat_fw_cu(
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);


torch::Tensor alpha_rat_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor alpha_rat,
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);

torch::Tensor weight_loss_fw_cu(
    const torch::Tensor weights,
    const torch::Tensor opacity,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);

torch::Tensor clear_weight_loss_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);

torch::Tensor nor_clear_weight_loss_fw_cu(
    const torch::Tensor opacity,
    const torch::Tensor c_sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);


torch::Tensor haz_clear_weight_loss_fw_cu(
    const torch::Tensor c_sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);



torch::Tensor haz_entropy_up_loss_fw_cu(
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);


torch::Tensor haz_entropy_up_loss_bw_cu(
    const torch::Tensor dL_dloss,
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
);