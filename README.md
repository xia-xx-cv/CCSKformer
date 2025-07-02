# CCSKformer
**ICME 2025 – Cross-Structure and Semantic Enhancement for Diabetic Retinopathy (DR) Grading**

## Prediction Instructions
1. Run `DataEnhanced.py` to perform two-stage data preprocessing. This will generate and save two versions of processed images along with their corresponding filenames.
2. Open `predict.py`, set the `DATA_PATH` in the `main()` function, and run the script to perform prediction.


## Notes
- Our model is inspired by [MVCINN](https://github.com/XiaolingLuo/MVCINN), where we followed the CNN-ViT hybrid structure but redesigned the basic blocks and introduced sub-band decomposition and KAN for enhanced representation.
- We partially initialized our model with their released weights, which may bring in some task-related prior knowledge, while some baselines were pretrained on ImageNet. We are trying to further analyze this and provide results trained from scratch in future updates.
- The current CS module computes dependencies **within each frequency**. We are working on enhancing this with **cross-frequency attention**.

## Acknowledgements
This repository is adapted from [MVCINN](https://github.com/XiaolingLuo/MVCINN). We thank the original authors for making their code publicly available:
```
@inproceedings{luo2023MVCINN,
  title     = {MVCINN: Multi-View Diabetic Retinopathy Detection Using a Deep Cross-Interaction Neural Network},
  author    = {Luo, Xiaoling and Liu, Chengliang and Wong, Waikeung and Wen, Jie and Jin, Xiaopeng and Xu, Yong},
  booktitle = {Thirty-Seventh AAAI Conference on Artificial Intelligence},
  year      = {2023}
}

## Citation
If you find our work somewhat helpful, please consider citing our paper:
```
@inproceedings{xia2025ccsk,
title={Cross-Structure and Semantic Enhancement for Diabetic Retinopathy Grading},
author={Xia, Xue and Lin, Zipeng and Zhu, Jingying and Yan, Jiebin and Fang, Yuming},
booktitle={The 2025 IEEE International Conference on Multimedia and Expo {(ICME)}},
year={2025},
}
