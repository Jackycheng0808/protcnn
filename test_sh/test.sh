python test.py \
        --model /home/workspace/protCNN/output/snapshots/resnet_test/epoch_2iter_1801.pth\
        --backbone resnet \
        --data_dir ./random_split \
        --train_dir train \
        --test_dir test \
        --batch_size 256 \
        --seq_max_len 120\
        --num_workers 8 \
