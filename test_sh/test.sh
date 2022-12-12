python test.py \
        --model /home/workspace/protCNN/output/snapshots/seNet_test/epoch_8iter_1001.pth\
        --backbone resnet \
        --data_dir ./random_split \
        --train_dir train \
        --test_dir test \
        --batch_size 512 \
        --seq_max_len 120\
        --num_workers 8 \
