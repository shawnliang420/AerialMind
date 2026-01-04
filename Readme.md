# AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios

This official repository of the paper [AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](https://arxiv.org/abs/2511.21053). 
<div align=center><img src="Figs/motivation.png"/></div>

## 📢 Latest Updates:

🔥🔥🔥**Dataset Now Publicly Available!**

Thank you everyone for your patience during the preparation phase. We are excited to announce that the dataset is now officially open for public use.

📥 **Baidu Netdisk:** [Baidu]()  
**Google Drive:** ⏳ *Upload in progress... (Links will be updated soon)*



## 💡 Building Your Own Dataset?
If you aim to construct a dataset similar to AerialMind, this repository [ **CRMOT**](https://github.com/chen-si-jia/CRMOT) serves as a comprehensive guide containing all the necessary resources and detailed pipeline information. We sincerely thank the authors for their contributions.

We referenced the [**RefDrone**](https://github.com/sunzc-sunny/refdrone) repository for the **COALA** methodology. We sincerely thank the authors for their contributions. We will release our core annotation tool (**Stage 2**) to facilitate future research.



# Dataset Features and Statistics
| Dataset        | Source       | Videos | Dom. | Reas. | Attr. | Expressions | Words | Instance / Expression | Instance | Annobbox  |
|----------------|--------------|--------|------|-------|-------|-------------|-------|-----------------------|----------|-----------|
| Refer-KITTI    | CVPR2023     | 18     | ✗    | ✗     | ✗     | 818         | 49    | 10.7                  | 8.8K     | 0.36M     |
| Refer-Dance    | CVPR2024     | 65     | ✗    | ✗     | ✗     | 1.9K        | 25    | 0.33                  | 650      | 0.55M     |
| Refer-KITTI-V2 | arXiv2024    | 21     | ✗    | ✗     | ✗     | 9.8K        | 617   | 6.7                   | 65.4K    | 3.06M     |
| Refer-UE-City  | arXiv2024    | 12     | ✗    | ✗     | ✗     | 714         | --    | 10.3                  | --       | 0.55M     |
| Refer-BDD      | IEEE TIM2025 | 50     | ✗    | ✗     | ✗     | 4.6K        | 225   | 14.1                  | 70.4K    | 1.50M     |
| CRTrack        | AAAI2025     | 41     | ✓    | ✗     | ✗     | 344         | 43    | --                    | --       | --        |
| LaMOT*         | IEEE ICRA2025| 62     | ✗    | ✗     | ✗     | 145         | 9     | **54.6**              | 508      | 1.2M      |
| AerialMind     | Ours         | **93** | ✓    | ✓     | ✓     | **24.6K**   | **1.2K** | 11.9              | **293.1K** | **46.14M** |

<div align=center><img src="Figs/dataset_analysis.png"/></div>



# Results
## Visualization
<div align=center><img src="Figs/vis.png"/></div>

# Getting started
## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        --  AerialMind
            |-- Attribute
            |-- image_02
                 -- Visdrone
                 -- UAVDT
            |-- labels_with_ids
   ```

## Installation

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

# TODO List

- [ ] release the dataset [Train]
- [ ] release the dataset [Test]

# Acknowledgements

📢We would like to express our sincere gratitude to the authors and developers of [ **RMOT**](https://github.com/wudongming97/RMOT)、[ **TempRMOT**](https://github.com/zyn213/TempRMOT)、[ **CRMOT**](https://github.com/chen-si-jia/CRMOT)、[ **RefDrone**](https://github.com/sunzc-sunny/refdrone). Their repository provided valuable guidance and inspiration for the construction ([ **CRMOT**](https://github.com/chen-si-jia/CRMOT)、[ **RefDrone**](https://github.com/sunzc-sunny/refdrone)) of our dataset. 

We also thank the community for their interest in AerialMind.







