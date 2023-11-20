# DCPNeRF
**Single-Scene Haze Removal and 3D Reconstruction Based on Dark Channel Prior**
![Overview of our method](https://github.com/C2022G/dcpnerf/blob/main/readme/method.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3


## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by
```python
git clone https://github.com/C2022G/dcpnerf.git
```
+  Create an anaconda environment
```python
conda create -n dcpnerf python=3.7
``` 
+ cuda code compilation dependency.
	- Install pytorch by
	```python
	conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
	```
	- Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) like
	```python
	pip install torch-scatter==2.0.9 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
	```
	- Install tinycudann following their [instrucion](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(pytorch extension) like
	```python
	pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
	```
	- Install apex following their [instruction](https://github.com/NVIDIA/apex#linux) like
	```python
	git clone https://github.com/NVIDIA/apex 
	cd apex 
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
	```
	- Install core requirements by
	```python
	pip install -r requirements.tx
	```
  
+ Cuda extension:please run this each time you pull the code.``.
 	```python
	pip install models/csrc/
	# (Upgrade pip to >= 22.1)
	```

## Datasets
Due to the lack of a dedicated single-haze scene image dataset, we employed virtual scenes as the experimental subjects.   We utilized Blender 3D models provided in NeRF to render realistic 360° panoramic images and depth maps while maintaining consistent camera poses and intrinsic parameters. Under the assumption of uniformly distributed haze particles in the virtual scenes, we endowed eight virtual scenes with uniform atmospheric light and the same haze density, achieved by applying fog to the rendered original clear images using the ASM formula.

**The dataset can be obtained from [Baidu net disk](https://pan.baidu.com/s/10vo99AKu6sAAfWD2ZYQL7w?pwd=2022) or [Google Drive](https://drive.google.com/file/d/1GeC3HEzEnf0yyYcxEUdlNLr1GDO6LbAD/view?usp=sharing)**


## Training
Our configuration file is available in config/opt.py, check it out for details.These profiles include the location of the dataset, the name of the experiment, the number of training sessions, and the loss function parameters. The above parameters may be different depending on the dataset.To train a Lego model, run

```python
python run.py  \
	--root_dir /devdata/chengan/Synthetic_NeRF/lego \
	--split train
	--exp_name lego_highter \
	--haz_dir_name highter \
	--num_epochs 5 \
	--composite_weight 1 \
	--distortion_weight 1e-3 \
	--opacity_weight 1e-3 \
	--dcp_weight 6e-3 \
	--foggy_weight 2e-4
```
After the training, three types of files will be generated in the current location, such as: log files are generated in the logs folder, model weight files are generated in ckpts, and rendering results are generated in the results folder, including test rendering images and videos of haze scenes and clean scenes

|dataset| dcp_weight | foggy_weight |
|--|--| --|
|  ficus | 2e-6 | 4e-6 |
|  mic | 6e-8 | 2e-6 |

If the default parameters are used in ficus and mic datasets, the haze component will occupy too much, resulting in missing object surfaces.This is shown in the figure below. Therefore, we reduce the intensity of the space occupation of haze particles.

![Overview of our method](https://github.com/C2022G/dcpnerf/blob/main/readme/over_occupancy.png)

Similarly, we can adjust dcp_weight and foggy_weight if the default parameters don't apply to a particular dataset.

If the scene is missing, dcp_weight and foggy_weight will be decreased, and if there are extra "tumors" in the scene, dcp_weight (mainly this one) and foggy_weight will be increased

**Similarly, when the scene haze concentration increases, the dcp_weight can be considered to increase.**

|Lego Haze concentration| dcp_weight | foggy_weight |
|--|--| --|
|  lower | 6e-3 | 2e-4 |
|  highter | 8e-2 | 2e-4 |

This is shown in the figure below.

![Overview of our method](https://github.com/C2022G/dcpnerf/blob/main/readme/haz_concentration.png)

## Visualization/Evaluation
By specifying the split, ckpt_path parameters, the run.py script supports rendering new camera trajectories, including test and val, from the pre-trained weights.To render test images,run

```python
python run.py  \
	--root_dir /devdata/chengan/Synthetic_NeRF/lego \
	--split test
	--exp_name lego_highter \
	--haz_dir_name highter 
	--ckpt_path /ckpts/..
```

## result
![Qualitative comparisons were performed on a synthesized hazy dataset.](https://github.com/C2022G/dcpnerf/blob/main/readme/result.png)


<table>
<thead>
  <tr>
    <th></th>
    <th colspan="2">Lego</th>
    <th colspan="2">Hotdog</th>
    <th colspan="2">Chair</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Method</td>
    <td>PSNR</td>
    <td>SSIM</td>
    <td>PSNR</td>
    <td>SSIM</td>
    <td>PSNR</td>
    <td>SSIM</td>
  </tr>
  <tr>
    <td>DCP+ngp</td>
    <td>23.90</td>
    <td>0.95</td>
    <td>19.60</td>
    <td><s>1.13</s></td>
    <td>23.30</td>
    <td><s>1.09</s></td>
  </tr>
  <tr>
    <td>WeatherDiffusion<br>+ngp</td>
    <td>20.30</td>
    <td><s>1.37</s></td>
    <td>20.80</td>
    <td><s>1.46</s></td>
    <td>22.10</td>
    <td><s>1.55</s></td>
  </tr>
  <tr>
    <td>FFANet+ngp</td>
    <td>22.50</td>
    <td>0.92</td>
    <td>23.30</td>
    <td>0.93</td>
    <td>21.90</td>
    <td>0.94</td>
  </tr>
  <tr>
    <td>DCPNeRF</td>
    <td>27.00</td>
    <td>0.94</td>
    <td>29.50</td>
    <td>0.959</td>
    <td>30.60</td>
    <td>0.972</td>
  </tr>
</tbody>
</table>


https://github.com/C2022G/dcpnerf/assets/151046579/e9f2b94e-8b70-4ca4-8152-8c5670e7ae4b


https://github.com/C2022G/dcpnerf/assets/151046579/10dd58b6-684c-4ce5-ac1e-46c2e1480b51


