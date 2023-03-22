# Style Uncertainty Based Self-Paced Meta Learning for Generalizable Person Re-Identification - Accepted at IEEE Transactions on Image Processing 2023

# Abstract
Domain generalizable person re-identification (DG ReID) is a challenging problem, because the trained model is often not generalizable to unseen target domains with different distribution from the source training domains. Data augmentation has been verified to be beneficial for better exploiting the source data to improve the model generalization. However, existing approaches primarily rely on pixel-level image generation that requires designing and training an extra generation network, which is extremely complex and provides limited diversity of augmented data. In this paper, we propose a simple yet effective feature based augmentation technique, named Style-uncertainty Augmentation (SuA). The main idea of SuA is to randomize the style of training data by perturbing the instance style with Gaussian noise during training process to increase the training domain diversity. And to better generalize knowledge across these augmented domains, we propose a progressive learning to learn strategy named Self-paced Meta Learning (SpML) that extends the conventional one-stage meta learning to multi-stage training process. The rationality is to gradually improve the model generalization ability to unseen target domains by simulating the mechanism of human learning. Furthermore, conventional person Re-ID loss functions are unable to leverage the valuable domain information to improve the model generalization. So we further propose a distance-graph alignment loss that aligns the feature relationship distribution among domains to facilitate the network to explore domain-invariant representations of images. Extensive experiments on four large-scale benchmarks demonstrate that our SuA-SpML achieves state-of-the-art generalization to unseen domains for person ReID.

# Requirements and Installation
CUDA>=10.0/
PyTorch 1.7.1/
python 3.6.10/
NumPy (1.16.4)/
TensorBoard/

# Description of the Code
main.py:driver script to run training/testing/


# Run
Train\
CUDA_VISIBLE_DEVICES=0,1,2 python main.py -a $ARCH --BNNeck --dataset_src1 $SRC1 --dataset_src2 $SRC2 --dataset_src3 $SRC3 -d $TARGET --logs-dir $LOG_DIR --data-dir $DATA_DIR

Evaluate\
python main.py \
-a $ARCH -d $TARGET \
--logs-dir $LOG_DIR --data-dir $DATA_DIR \
--evaluate --resume $RESUME

Training Data

The model is trained and evaluated on Market-1501, DukeMTMC-reID, MSMT17, CUHK03
