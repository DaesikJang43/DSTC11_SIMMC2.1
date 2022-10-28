# DSTC11_SIMMC2.1
The repository to participate the DSTC11 Track 1 SIMMC 2.1

## Overview
- This model is for DSTC11 Track1. Sub-Task 1 model is based on CoCondenser and Vistion Transformer(ViT) which has a powerful enocding ablity. 
- The final submission files are in the "final" branch
---
## Enviroment

- CUDA 11.1
- Python 3.8+

Packages:
- torch==1.12.1
- transformers==4.21.1
- scikit-learn==1.1.2
- tensorboard==2.10.0
---

## Model Parameters
Model Parameters are shared by Google Drive.

- Download the Fine-tuned Disambiguate Model    
Download Link: [Download](https://drive.google.com/file/d/1NrBXwR5Ny4ceMXNrQTgkBpd8X2bVZcdG/view?usp=sharing)    
Download Path : disambiguate/outputs    

- Download the Fine-tuned Image Classification Model for fasion    
Download Link: [Download](https://drive.google.com/file/d/1oYHfl_9zm3_IT3b9ypytomw1aO3uZsjY/view?usp=sharing)    
Download Path : image_classification/outputs/fashion    

- Download the Fine-tuned Image Classification Model for furniture    
Download Link: [Download](https://drive.google.com/file/d/1o_TdfE-mSIJQY25aRsgBSYKOlYL9BObw/view?usp=sharing)    
Download Path : image_classification/outputs/furniture    

- Download the Fine-tuned Ambiguous Candidates Model    
Download Link: [Download](https://drive.google.com/file/d/1oVAMRs9iGKufmk-lzFC6nJUzFDbXRtIS/view?usp=sharing)    
Download Path : ambiguous_candidates/outputs    

## Processing

### 1. Preprocessing for ambiguous candidates
```sh
# (train/dev/devtet)
python format_ambiguous_candidates_data.py \
	--simmc_train_json ./data/simmc2.1_dials_dstc11_train.json \
	--simmc_dev_json ./data/simmc2.1_dials_dstc11_dev.json \
	--simmc_devtest_json ./data/simmc2.1_dials_dstc11_devtest.json \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_public/ \
	--data_mode public \
	--ambiguous_candidates_save_path ./data

# (teststd)
python format_ambiguous_candidates_data.py \
	--simmc_teststd_json ./data/simmc2.1_dials_dstc11_teststd_public.json \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_teststd \
	--data_mode teststd \
	--ambiguous_candidates_save_path ./data
```

### 2. Preprocessing for ambiguous candidates image features

```sh
# (train/dev/devtet)
python ambiguous_candidates_image_preprocess.py \
	--data_folder ./data \
	--data_mode public \
	--scene_image_folder ./simmc2_scene_images_dstc10_public \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_public \
	--save_folder ./data/image_features

# (teststd)
python ambiguous_candidates_image_preprocess.py \
	--data_folder ./data \
	--data_mode teststd \
	--scene_image_folder ./simmc2_scene_images_dstc10_teststd \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_teststd \
	--save_folder ./data/image_features
```

### 3. Preprocessing for visual metadata classification

```sh
# (train/dev/devtet)
python ImageClassification_preprocess.py \
	--data_folder ./data \
	--data_mode public \
	--save_folder ./data/image_classification \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_public

# (teststd)
python ImageClassification_preprocess.py \
	--data_folder ./data \
	--data_mode teststd \
	--save_folder ./data/image_classification \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_teststd
```

### 4. Training and inference for visual metadata classification

```sh
# (fashion train/dev/devtet)
python main.py \
	--do_train \
	--do_eval \
	--data_dir ../data/image_classification  \
	--task 1234 \
	--learning_rate 9e-5 \
	--num_train_epochs 10 \
	--vit_model google/vit-large-patch32-384 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--output_dir ./outputs \
	--domain fashion

# (furniture train/dev/devtet)
python main.py \
	--do_train \
	--do_eval \
	--data_dir ../data  \
	--task 12 \
	--learning_rate 9e-5 \
	--num_train_epochs 10 \
	--vit_model google/vit-large-patch32-384 \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--output_dir ./outputs \
	--domain furniture

# (fashion inference)
python main.py \
	--do_eval \
	--submission \
	--data_dir ../data  \
	--task 1234 \
	--eval_batch_size 16 \
	--output_dir ./outputs/fashion \
	--domain fashion

# (furniture inference)
python main.py \
	--do_eval \
	--submission \
	--data_dir ../data  \
	--task 12 \
	--eval_batch_size 16 \
	--output_dir ./outputs/furniture \
	--domain furniture
```

### 5. Preprocessing for ambiguous candidates visual metadata features

```sh
# (train)
python visual_metadata_preprocess.py \
	--metadata_folder ./data \
	--data_mode train \
	--simmc_train_json ./data/simmc2.1_dials_dstc11_train.json \
	--scene_json_folder ./simmc2_scene_jsons_dstc10_public \
	--save_folder ./data/image_features

# (dev/devtest/teststd)
python visual_metadata_preprocess.py \
	--data_mode predicted_data \
	--predicted_json_folder ./data/image_classification \
	--save_folder ./data/./data/image_features
```

### 6. Training and inference for disambiguate

```sh
# (train/dev/devtet)
python main.py \
	--do_train \
	--do_eval \
	--data_dir ../data  \
	--learning_rate 1e-5 \
	--num_train_epochs 10 \
	--model_name Luyu/co-condenser-wiki \
	--train_batch_size 16 \
	--eval_batch_size 16 \
	--output_dir ./outputs 

# (teststd)
python main.py \
	--do_eval \
	--submission \
	--data_dir ../data  \
	--eval_batch_size 16 \
	--output_dir ./outputs 
```

### 7. Training and inference for ambiguous candidates

```sh
# (train/dev/devtet)
python main.py \
	--do_train \
	--do_eval \
	--data_dir ../data  \
	--learning_rate 5e-5 \
	--num_train_epochs 10 \
	--bert_model Luyu/co-condenser-wiki \
	--vit_model google/vit-base-patch16-224-in21k \
	--max_turns 3 \
	--negative_sample_size 10 \
	--positive_weight 3.0 \
	--temperature 1.0 \
	--train_batch_size 16 \
	--output_dir ./outputs 

# (teststd)
python main.py \
	--do_eval \
	--submission \
	--data_dir ../data  \
	--output_dir ./outputs 
```

## Final result
The final prediction files that we submitted are located in the path below. These are the final prediction results intended to be compared to those of other models

```sh
ambiguous_candidates/outputs/dstc11-simmc-teststd-pred-subtask-123.json
```

## devtest Performance
|  Sub-Task #1   | Recall | Precision |    F1    |
| :------------: | :----: | :-------: | :------: | 
|  Our Model     |  76.01 |   63.64   |   69.28  |