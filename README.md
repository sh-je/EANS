
# EANS-PyTorch

This code is PyTorch implementation for the our paper, [Entity Aware Negative Sampling with Auxiliary Loss of False Negative Prediction for Knowledge Graph Embedding](https://arxiv.org/abs/2210.06242).

## Environments and Requirements

- Python >= 3.6
- PyTorch >= 1.0.0
- A Nvidia GPU with CUDA 10.0+

All required libraries can be installed by the command below.

    pip install -r requirements.txt

## Datasets

Our project use two standard benchmark, FB15K-237 and WN18RR, the most widely used in link prediction task. 
The datasets are placed in `./data` directory.
Each dataset contains its own vocabulary files and triple files for train, valid and test.
The statistics of two datasets are summarized in table below.

| Dataset | #entity | #relation | #train | #valid | #test |
|----|----|----|----|----|----|
| FB15K-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |


## Model Training
The example scripts are located in `./scripts` directory. You can copy them and modify for your experiments. 
For example, these codes are in `./scripts/train_transe_fb15k237.sh`, which train a TransE model with EANS on FB15K-237 dataset.
```
...
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u codes/run.py 
    --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -g $GAMMA -a $ALPHA -r $REGULARIZATION \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    -sm $SAMPLING_METHOD \
    -k $NCLUSTERS  -v $VARIANCE --reorder_steps $REORDER_STEPS \
    -sub -subl $SUB_LOSS_WEIGHT -subr $SUB_REGULARIZATION \
...
```

## Reproducing the Results

The best configurations for EANS are defined in bash scripts under the `./scripts` directory. 
To reproduce the results in the [paper](https://), you can copy the scripts and just run the codes.

For example, the best performance of TransE with EANS on FB15K-237 can be reproduced by these commands.
```shell
# copy the script containing best hyperparams
cp ./scripts/train_transe_fb15k237.sh ./
# run the script 
bash train_transe_fb15k237.sh
```

We have prepared each shell script in advance for all scoring models and datasets presented in the paper.

## Results of EANS

### FB15K-237
|  | TransE | TransD | DistMult | ComplEx | RotatE |
|----|----|----|----|----|----|
| MRR | .342 | .340 | .309 | .323 | .344 |
| MR | 172 | 184 | 397 | 454 | 165 |
| HITS@1 | .243 | .243 | .222 | .234 | .247 |
| HITS@3 | .380 | .380 | .340 | .356 | .381 |
| HITS@10 | .534 | .534 | .482 | .503 | .537 |
| Time | 2h 58m | 6h 30m | 4h 52m | 5h 57m | 7h 49m |

### WN18RR
|  | TransE | TransD | DistMult | ComplEx | RotatE |
|----|----|----|----|----|----|
| MRR | .228 | .225 | .438 | .463 | .489 |
| MR | 3686 | 6640 | 4938 | 5350 | 3402 |
| HITS@1 | .026 | .041 | .391 | .417 | .447 |
| HITS@3 | .398 | .407 | .454 | .484 | .504 |
| HITS@10 | .533 | .491 | .537 | .558 | .576 |
| Time | 3h 26m | 5h 40m | 5h 23m | 6h 33m | 9h 13m |

The training times are measured on a single Nvidia V100 GPU machine.

## Citation

If you use the codes, please cite our [paper](https://arxiv.org/abs/2210.06242):

```
@article{je2022entity,
  title={Entity Aware Negative Sampling with Auxiliary Loss of False Negative Prediction for Knowledge Graph Embedding},
  author={Je, Sang-Hyun},
  journal={arXiv preprint arXiv:2210.06242},
  year={2022}
}
```

## References

This code is based on the [RotatE](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) project.
The basic training process and KGE models codes are copied from the project.
