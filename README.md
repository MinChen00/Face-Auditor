## Introduction

This repo contains the implementation of Face-Auditor, which aims to evaluate the privacy leakage in the state-of-the-art few-shot learning pipeline.

### Code Strcuture

```
.
├── config.py
├── exp
├── lib_classifer
├── lib_dataset
├── lib_metrics
├── lib_model
├── main.py
├── parameter_parser.py
└── README.md
```

### Environment Prepare

```bash
conda create --name face_auditor python=3.6.10
conda activate face_auditor
pip install numpy pandas seaborn matplotlib sklearn MulticoreTSNE cython facenet_pytorch deepface opacus psutil GPUtil
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Dataset Prepare

The corresponding file for downloading the datasets is in `lib_dataset/datasets/`.
In our paper, we mainly focus on four open-source human face image datasets, they are:

- UMDFaces
- WebFace
- VGGFace2
- CelebA

Other datasets should also work with our Face-Auditor.

### Evaluations
In the following, we give some examples of the experimental configurations, see more details in `parameter_parser.py`.


#### Training Shadow and Target Models
```
exp='class_mem_infer_meta'

python main.py --exp $exp --is_train_target true --is_train_shadow true
```

#### Constructing the Probing Set
```
shot=5
way=5
probe_num_task=100
probe_num_query=5

python main.py --is_generate_probe true --probe_ways $way --probe_shot $shot --probe_num_task $probe_num_task --probe_num_query $probe_num_query 
```

#### Reference Information related Configurations
```
## probe controlling parameters ##
python main.py --is_similarity_aided true --is_use_image_similarity true --image_similarity_name cosine

```


#### On the Robustness of FACE-AUDITOR

```

## adv (input) defense parameters ##
python main.py --is_adv_defense true

## dp (training) defense parameters ##
python main.py --is_dp_defense true

## noise (output) defense parameters ##
python main.py --is_noise_defense true

## memguard (adaptive) defense parameters ##
python main.py --is_memguard_defense true

```