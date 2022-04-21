# FastRE
The code and dataset for "FastRE: Towards Fast Relation Extraction with Convolutional Encoder and Improved Cascade Binary Tagging Framework"

### Requirements
This repository is tested on PaddlePaddle==2.2.0 with CUDA==10.2 and cuDNN==7.6. Normally, the following environments are required:
- python 3.7 +
- paddlepaddle-gpu 1.8 +
- numpy 1.19 +
- tqdm

### Usage
By default, use the following commands to train the model for 60 epochs, save the model with the best performance, and finally test it on the test set.

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
                --device gpu_num
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
                --device gpu_num
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
                --device gpu_num
```

### Acknowledgement
We refer to the code of [CASREL](https://github.com/weizhepei/CasRel). Thanks for their contributions.

### Reference
Please cite our paper if you find our work useful for your research:
```
@inproceedings{li2022FastRE,
  title={FastRE: Towards Fast Relation Extraction with Convolutional Encoder and Improved Cascade Binary Tagging Framework},
  author={Li, Guozheng and Chen, Xu and Wang, Peng and Xie, Jiafeng and Luo, Qiqing},
  booktitle={Proceedings of the 31st International Joint Conference On Artificial Intelligence},
  year={2022}
  ```
}
