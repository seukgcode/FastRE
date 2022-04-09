# FastRE
The code and dataset for "FastRE: Towards Fast Relation Extraction with Convolutional Encoder and Improved Cascade Binary Tagging Framework"

### Requirements
This repository is tested on PaddlePaddle==2.2.0 with CUDA==10.2 and cuDNN==7.6. Normally, the following environments are required:
- python 3.7 +
- paddlepaddle-gpu 1.8 +
- numpy 1.19 +
- tqdm

### Usage
- Train and test on NYT10
```
python train.py --name NYT10 \
                --train_path ./data/nyt10/new_train.json \
                --valid_path ./data/nyt10/new_valid.json \
                --test_path ./data/nyt10/new_test.json \
                --schemas_path ./auxiliary/nyt10_schemas.json \
                --num_relations 29 \
                --num_subs 4 \
                --num_objs 4 \
                --device cpu
```

- Train and test on NYT11
```
python train.py --name NYT11 \
                --train_path ./data/nyt11/new_train.json \
                --valid_path ./data/nyt11/new_valid.json \
                --test_path ./data/nyt11/new_test.json \
                --schemas_path ./auxiliary/nyt11_schemas.json \
                --num_relations 12 \
                --num_subs 3 \
                --num_objs 3 \
                --device cpu
```
- Train and test on NYT24
```
python train.py --name NYT24 \
                --train_path ./data/nyt24/new_train.json \
                --valid_path ./data/nyt24/new_valid.json \
                --test_path ./data/nyt24/new_test.json \
                --schemas_path ./auxiliary/nyt24_schemas.json \
                --num_relations 24 \
                --num_subs 4 \
                --num_objs 4 \
                --device cpu
```

### Acknowledgement
We refer to the code of [CASREL](https://github.com/weizhepei/CasRel). Thanks for their contributions.
