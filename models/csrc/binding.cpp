#include "utils.h"


std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor half_sizes,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(half_sizes);
    return ray_aabb_intersect_cu(rays_o, rays_d, centers, half_sizes, max_hits);
}


std::vector<torch::Tensor> ray_sphere_intersect(
    const torch::Tensor rays_o,
    const torch::Tensor rays_d,
    const torch::Tensor centers,
    const torch::Tensor radii,
    const int max_hits
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(centers);
    CHECK_INPUT(radii);
    return ray_sphere_intersect_cu(rays_o, rays_d, centers, radii, max_hits);
}


void packbits(
    torch::Tensor density_grid,
    const float density_threshold,
    torch::Tensor density_bitfield
){
    CHECK_INPUT(density_grid);
    CHECK_INPUT(density_bitfield);

    return packbits_cu(density_grid, density_threshold, density_bitfield);
}


torch::Tensor morton3D(const torch::Tensor coords){
    CHECK_INPUT(coords);

    return morton3D_cu(coords);
}


torch::Tensor morton3D_invert(const torch::Tensor indices){
    CHECK_INPUT(indices);

    return morton3D_invert_cu(indices);
}


std::vector<torch::Tensor> raymarching_train(
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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(density_bitfield);
    CHECK_INPUT(noise);

    return raymarching_train_cu(
        rays_o, rays_d, hits_t, density_bitfield, cascades,
        scale, exp_step_factor, noise, grid_size, max_samples);
}


std::vector<torch::Tensor> raymarching_test(
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
){
    CHECK_INPUT(rays_o);
    CHECK_INPUT(rays_d);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(density_bitfield);

    return raymarching_test_cu(
        rays_o, rays_d, hits_t, alive_indices, density_bitfield, cascades,
        scale, exp_step_factor, grid_size, max_samples, N_samples);
}



std::vector<torch::Tensor> distortion_loss_fw(
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_fw_cu(ws, deltas, ts, rays_a);
}


torch::Tensor distortion_loss_bw(
    const torch::Tensor dL_dloss,
    const torch::Tensor ws_inclusive_scan,
    const torch::Tensor wts_inclusive_scan,
    const torch::Tensor ws,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a
){
    CHECK_INPUT(dL_dloss);
    CHECK_INPUT(ws_inclusive_scan);
    CHECK_INPUT(wts_inclusive_scan);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(rays_a);

    return distortion_loss_bw_cu(dL_dloss, ws_inclusive_scan, wts_inclusive_scan,
                                 ws, deltas, ts, rays_a);
}




std::vector<torch::Tensor> haz_composite_train_fw(
    const torch::Tensor sigmas,
    const torch::Tensor a_sigmas,
    const torch::Tensor rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(sigmas);
    CHECK_INPUT(a_sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);

    return haz_composite_train_fw_cu(
                sigmas,a_sigmas, rgbs, deltas,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> haz_composite_train_bw(
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
){
    CHECK_INPUT(dL_dopacity);
    CHECK_INPUT(dL_drgb);
    CHECK_INPUT(dL_dws);
    CHECK_INPUT(sigmas);
    CHECK_INPUT(a_sigmas);
    CHECK_INPUT(rgbs);
    CHECK_INPUT(ws);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(opacity);
    CHECK_INPUT(rgb);

    return haz_composite_train_bw_cu(
                dL_dopacity, dL_drgb, dL_dws,
                sigmas,a_sigmas, rgbs, ws, deltas, rays_a,
                opacity, rgb, opacity_threshold);
}




std::vector<torch::Tensor> all_composite_train_fw(
    const torch::Tensor c_sigmas,
    const torch::Tensor p_sigmas,
    const torch::Tensor c_rgbs,
    const torch::Tensor p_rgbs,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const float opacity_threshold
){
    CHECK_INPUT(c_sigmas);
    CHECK_INPUT(p_sigmas);
    CHECK_INPUT(c_rgbs);
    CHECK_INPUT(p_rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);

    return all_composite_train_fw_cu(
                c_sigmas,p_sigmas, c_rgbs,p_rgbs, deltas,
                rays_a, opacity_threshold);
}


std::vector<torch::Tensor> all_composite_train_bw(
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
){
    CHECK_INPUT(dL_df_rgb)
    CHECK_INPUT(dL_df_opacity)
    CHECK_INPUT(c_sigmas)
    CHECK_INPUT(p_sigmas)
    CHECK_INPUT(c_rgbs)
    CHECK_INPUT(p_rgbs)
    CHECK_INPUT(deltas)
    CHECK_INPUT(rays_a)
    CHECK_INPUT(f_rgb)
    CHECK_INPUT(f_opacity)

    return all_composite_train_bw_cu(
                dL_df_rgb,dL_df_opacity,
                c_sigmas,p_sigmas,
                c_rgbs,p_rgbs,
                deltas, rays_a,
                f_rgb,f_opacity,
                T_threshold);
}


void all_composite_test_fw(
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
){
    CHECK_INPUT(c_sigmas);
    CHECK_INPUT(p_sigmas);
    CHECK_INPUT(c_rgbs);
    CHECK_INPUT(p_rgbs);
    CHECK_INPUT(deltas);
    CHECK_INPUT(ts);
    CHECK_INPUT(hits_t);
    CHECK_INPUT(alive_indices);
    CHECK_INPUT(N_eff_samples);
    CHECK_INPUT(c_rgb);
    CHECK_INPUT(c_rgb);
    CHECK_INPUT(c_opacity);
    CHECK_INPUT(f_rgb);
    CHECK_INPUT(f_rgb);
    CHECK_INPUT(f_opacity);

    return all_composite_test_fw_cu(
                c_sigmas,p_sigmas, c_rgbs,p_rgbs,
                deltas,ts,hits_t,alive_indices,T_threshold,N_eff_samples,c_rgb,c_depth,c_opacity,f_rgb,f_depth,f_opacity);
}




torch::Tensor foggy_fw(
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    CHECK_INPUT(weights);
    CHECK_INPUT(alpha_rat);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(vr_samples);

    return foggy_fw_cu(weights,alpha_rat,rays_a,vr_samples);
}


torch::Tensor foggy_bw(
    const torch::Tensor dL_dloss,
    const torch::Tensor weights,
    const torch::Tensor alpha_rat,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    CHECK_INPUT(dL_dloss);
    CHECK_INPUT(weights);
    CHECK_INPUT(alpha_rat);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(vr_samples);


    return foggy_bw_cu(dL_dloss,weights,alpha_rat,rays_a,vr_samples);
}




torch::Tensor alpha_rat_fw(
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){

    CHECK_INPUT(sigma);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(vr_samples);

    return alpha_rat_fw_cu(sigma,deltas,rays_a,vr_samples);
}


torch::Tensor alpha_rat_bw(
    const torch::Tensor dL_dloss,
    const torch::Tensor alpha_rat,
    const torch::Tensor sigma,
    const torch::Tensor deltas,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    CHECK_INPUT(dL_dloss);
    CHECK_INPUT(alpha_rat);
    CHECK_INPUT(sigma);
    CHECK_INPUT(deltas);
    CHECK_INPUT(rays_a);
    CHECK_INPUT(vr_samples);


    return alpha_rat_bw_cu(dL_dloss,alpha_rat,sigma,deltas,rays_a,vr_samples);
}



torch::Tensor haz_clear_weight_loss_fw(
    const torch::Tensor c_sigmas,
    const torch::Tensor deltas,
    const torch::Tensor ts,
    const torch::Tensor rays_a,
    const torch::Tensor vr_samples
){
    CHECK_INPUT(c_sigmas)
    CHECK_INPUT(deltas)
    CHECK_INPUT(ts)
    CHECK_INPUT(rays_a)
    CHECK_INPUT(vr_samples)

    return haz_clear_weight_loss_fw_cu(c_sigmas,deltas,ts,rays_a,vr_samples);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("ray_aabb_intersect", &ray_aabb_intersect);
    m.def("ray_sphere_intersect", &ray_sphere_intersect);

    m.def("morton3D", &morton3D);
    m.def("morton3D_invert", &morton3D_invert);
    m.def("packbits", &packbits);

    m.def("raymarching_train", &raymarching_train);
    m.def("raymarching_test", &raymarching_test);


    m.def("distortion_loss_fw", &distortion_loss_fw);
    m.def("distortion_loss_bw", &distortion_loss_bw);


    m.def("haz_composite_train_fw", &haz_composite_train_fw);
    m.def("haz_composite_train_bw", &haz_composite_train_bw);

    m.def("all_composite_train_fw", &all_composite_train_fw);
    m.def("all_composite_train_bw", &all_composite_train_bw);

    m.def("all_composite_test_fw", &all_composite_test_fw);

    m.def("foggy_fw", &foggy_fw);
    m.def("foggy_bw", &foggy_bw);

    m.def("alpha_rat_fw", &alpha_rat_fw);
    m.def("alpha_rat_bw", &alpha_rat_bw);

    m.def("haz_clear_weight_loss_fw", &haz_clear_weight_loss_fw);

}