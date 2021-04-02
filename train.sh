#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,2,3,7

#--------------------------------------------------------------------------------------------------
# Pretrain the model on LVIS dataset
./tools/dist_train.sh configs/baselines/faster_rcnn_r50_fpn_1x_lvis.py 8 

# test
./tools/dist_test_lvis.sh configs/baselines/faster_rcnn_r50_fpn_1x_lvis.py work_dirs/baselines/faster_rcnn_r50_fpn_1x_lvis/epoch_12.pth 8 --out work_dirs/baselines/faster_rcnn_r50_fpn_1x_lvis/val_results.pkl --eval bbox

#--------------------------------------------------------------------------------------------------
# Finetune the classifier
./tools/dist_train.sh configs/acsl/faster_rcnn_r50_fpn_1x_lvis_tunefc_acsl.py 8

# test
./tools/dist_test_lvis.sh configs/acsl/faster_rcnn_r50_fpn_1x_lvis_tunefc_acsl.py work_dirs/acsl/faster_rcnn_r50_fpn_1x_lvis_tunefc_acsl/epoch_12.pth 8 --out work_dirs/acsl/faster_rcnn_r50_fpn_1x_lvis_tunefc_acsl/val_results_final.pkl --eval bbox

