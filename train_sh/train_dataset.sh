python train.py \
        --output_version mobilenet_slim\
        --data_dir ./random_split \
        --train_dir train \
        --val_dir val \
        --num_epochs 100 \
        --batch_size 256 \
        --backbone MobileNetV2 \
        --optimizer adam \
        --early_stop False \
        --lr 0.001 \
        --weight_decay 0 \
        --loss_function categorical \
        --num_workers 8 \

