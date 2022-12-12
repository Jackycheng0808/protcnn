python train.py \
        --output_version seNet_test_adam_0\
        --data_dir ./random_split \
        --train_dir train \
        --val_dir val \
        --num_epochs 10 \
        --batch_size 512 \
        --backbone resSEnet \
        --optimizer adam \
        --early_stop False \
        --lr 0.0001 \
        --weight_decay 0 \
        --loss_function categorical \
        --num_workers 8 \

