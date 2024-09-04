Data Preparation and Evaluation for POPE
---

Here we take OpenImages (oi) as an example, and you can also use it on AMBER in similar ways.

# Data Preparation for POPE
We provide our preprocessed data for POPE in `output/`. If you want to prepare the data by yourself, follow the steps below.

First, make sure that you have completed the data preparation process for Pfram(O) in [../README.md](../README.md).

1. Convert object annotations to POPE format:
```shell
python create_segmentation.py --dataset oi \
--input_fname ../output/oi/anno_dict.json \
--output_fname segmentation/oi_ground_truth_segmentation.json
```

1. Create POPE questions:
```shell
python main.py --dataset oi \
--seg_path segmentation/oi_ground_truth_segmentation.json
```

After running the above steps, you will get the annotations and questions of OpenImages in `output/oi` folder. This folder should contain the following files:
- `oi_ground_truth_objects.json` and `oi_co_occur.json`: files related to object annotations.
- `oi_pope_random.json`, `oi_pope_popular.json` and `oi_pope_adversarial.json`: Files that contain POPE questions with different negative sampling methods. See [POPE paper] for details.
- `oi_pope_questions.json`: As the above 3 question files with different negative sampling methods share the same positive questions, this file contains questions after deduplication. You can simply evaluate the models with questions in this file as input.


# Inference and Evaluation on POPE
We take InstructBLIP as an example, while you can also use other models.

1. Model inference on POPE.
```shell
python inference.py \
--img_folder OPENIMAGES_IMAGE_FOLDER \
--input output/oi/oi_pope_questions.json \
--model_name Salesforce/instructblip-vicuna-7b \
--output output/oi/instructblip-vicuna-7b/oi_pope_questions.json
```

1. Split the inference output into 3 files with different negative sampling (random, popular and adversarial):
```shell
# python split_results.py MODEL_NAME DATASET_NAME
python split_results.py instructblip-vicuna-7b oi
```

1. Check the accuracy for each negative sampling file:
```shell
python evaluate.py results/oi/instructblip-vicuna-7b/oi_pope_%s.json output/oi/instructblip-vicuna-7b/oi_pope_%s.json
```