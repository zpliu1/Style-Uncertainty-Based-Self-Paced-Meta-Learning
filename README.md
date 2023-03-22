# Style-uncertainty Based Self-paced Meta Learning/
Style Uncertainty Based Self-Paced Meta Learning for Generalizable Person Re-Identification

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
