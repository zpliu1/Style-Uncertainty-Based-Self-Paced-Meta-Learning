# SuA-SpML
Style Uncertainty Based Self-Paced Meta Learning for Generalizable Person Re-Identification

#Train
CUDA_VISIBLE_DEVICES=0,1,2 python main.py -a $ARCH --BNNeck --dataset_src1 $SRC1 --dataset_src2 $SRC2 --dataset_src3 $SRC3 -d $TARGET --logs-dir $LOG_DIR --data-dir $DATA_DIR

# evaluate
python main.py \
-a $ARCH -d $TARGET \
--logs-dir $LOG_DIR --data-dir $DATA_DIR \
--evaluate --resume $RESUME
