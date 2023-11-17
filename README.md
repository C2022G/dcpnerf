# DCPNeRF
**Single-Scene Haze Removal and 3D Reconstruction Based on Dark Channel Prior**
![Overview of our method](https://github.com/C2022G/dcpnerf/blob/main/readme/method.png)

The implementation of our code is referenced in [kwea123-npg_pl](https://github.com/kwea123/ngp_pl)。The hardware and software basis on which our model operates is described next
 - Ubuntu 18.04
 -  NVIDIA GeForce RTX 3090 ,CUDA 11.3


## Setup
Let's complete the basic setup before we run the model。

 
+ Clone this repo by `git clone https://github.com/C2022G/dcpnerf.git`
+  Create an anaconda environment `conda create -n dcpnerf python=3.7` 
+ cuda code compilation dependency.
	- Install pytorch by `conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch`
	- Install torch-scatter following their [instruction](https://github.com/rusty1s/pytorch_scatter#installation) like `pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html`
	- Install tinycudann following their [instrucion](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension)(pytorch extension)
	- Install apex following their [instruction](https://github.com/NVIDIA/apex#linux)
	- Install core requirements by `pip install -r requirements.txt`
+ Cuda extension:please run this each time you pull the code.`pip install models/csrc/`.(Upgrade pip to >= 22.1)
 

## Datasets
Due to the lack of a dedicated single-haze scene image dataset, we employed virtual scenes as the experimental subjects.   We utilized Blender 3D models provided in NeRF to render realistic 360° panoramic images and depth maps while maintaining consistent camera poses and intrinsic parameters. Under the assumption of uniformly distributed haze particles in the virtual scenes, we endowed eight virtual scenes with uniform atmospheric light and the same haze density, achieved by applying fog to the rendered original clear images using the ASM formula.

**The dataset can be obtained from Baidu net disk:**
链接: https://pan.baidu.com/s/1onhDndrbwA39h3ee07iYwA 	
提取码: 2022

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



https://github.com/C2022G/dcpnerf/assets/151046579/e9f2b94e-8b70-4ca4-8152-8c5670e7ae4b


https://github.com/C2022G/dcpnerf/assets/151046579/10dd58b6-684c-4ce5-ac1e-46c2e1480b51


