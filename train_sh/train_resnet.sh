python train.py \
        --output_version resnet_test_2 \
        --data_dir ./random_split \
        --train_dir train \
        --val_dir val \
        --num_epochs 10 \
        --batch_size 512 \
        --backbone resnet \
        --optimizer adam \
        --early_stop False \
        --lr 0.0001 \
        --weight_decay 0.0005 \
        --loss_function categorical \
        --seq_max_len 120 \
        --num_workers 8 \

