python train.py \
        --output_version seNet_\
        --data_dir ./random_split \
        --train_dir train \
        --val_dir val \
        --num_epochs 10 \
        --batch_size 256 \
        --backbone resSEnet \
        --optimizer RAdam \
        --early_stop False \
        --lr 0.005 \
        --weight_decay 0 \
        --loss_function categorical \
        --num_workers 8 \

