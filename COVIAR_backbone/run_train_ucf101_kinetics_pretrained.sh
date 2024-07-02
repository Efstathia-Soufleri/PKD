# SPLIT1
# train residual rolled model pretrained on kinetics residual
for seed in 4711
do
python train_network_pretrained_kinetics_residual_init_R.py --lr 0.005 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation residual \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split1_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split1_test.txt \
--model-prefix kinetics_pretrained_residual_ucf101_residual_rolled_model_resnet18_seed_${seed}_split1_v2 \
--lr-steps 150 270 390 --epochs 510 --split split1 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train mv rolled model on SPLIT1 pretrained on kinetics mv
for seed in 4711
do
python train_network_pretrained_kinetics_mv.py --lr 0.01 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation mv \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split1_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split1_test.txt \
--model-prefix kinetics_pretrained_mv_ucf101_mv_rolled_model_resnet18_seed_${seed}_split1_v2 \
--lr-steps 150 270 390 --epochs 510 --split split1 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train iframe rolled model resnet50 pretrained on kinetics
for seed in 4711
do
python train_network_pretrained_kinetics.py --lr 0.0003 --batch-size 40 --arch resnet50 \
--data-name ucf101 --representation iframe \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split1_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split1_test.txt \
--model-prefix kinetics_pretrained_ucf101_iframe_rolled_model_resnet50_seed_${seed}_split1_v3 \
--lr-steps 150 270 390 --epochs 510 --split split1 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

#######################################################################

# SPLIT2
# train residual rolled model pretrained on kinetics residual
for seed in 4711
do
python train_network_pretrained_kinetics_residual_init_R.py --lr 0.005 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation residual \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split2_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split2_test.txt \
--model-prefix kinetics_pretrained_residual_ucf101_residual_rolled_model_resnet18_seed_${seed}_split2_v2 \
--lr-steps 150 270 390 --epochs 510 --split split2 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train mv rolled model on SPLIT2 pretrained on kinetics mv
for seed in 4711
do
python train_network_pretrained_kinetics_mv.py --lr 0.01 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation mv \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split2_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split2_test.txt \
--model-prefix kinetics_pretrained_mv_ucf101_mv_rolled_model_resnet18_seed_${seed}_split2_v2 \
--lr-steps 150 270 390 --epochs 510 --split split2 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train iframe rolled model resnet50 pretrained on kinetics
for seed in 4711
do
python train_network_pretrained_kinetics.py --lr 0.0003 --batch-size 40 --arch resnet50 \
--data-name ucf101 --representation iframe \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split2_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split2_test.txt \
--model-prefix kinetics_pretrained_ucf101_iframe_rolled_model_resnet50_seed_${seed}_split2_v3 \
--lr-steps 150 270 390 --epochs 510 --split split2 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

######################################################################################
# SPLIT3
# train residual rolled model pretrained on kinetics residual
for seed in 4711
do
python train_network_pretrained_kinetics_residual_init_R.py --lr 0.005 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation residual \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split3_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split3_test.txt \
--model-prefix kinetics_pretrained_residual_ucf101_residual_rolled_model_resnet18_seed_${seed}_split3_v2 \
--lr-steps 150 270 390 --epochs 510 --split split3 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train mv rolled model on SPLIT2 pretrained on kinetics mv
for seed in 4711
do
python train_network_pretrained_kinetics_mv.py --lr 0.01 --batch-size 80 --arch resnet18 \
--data-name ucf101 --representation mv \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split3_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split3_test.txt \
--model-prefix kinetics_pretrained_mv_ucf101_mv_rolled_model_resnet18_seed_${seed}_split3_v2 \
--lr-steps 150 270 390 --epochs 510 --split split3 \
--gpus 0 1 --random_seed $seed --optimizer adam
done

# train iframe rolled model resnet50 pretrained on kinetics
for seed in 4711
do
python train_network_pretrained_kinetics.py --lr 0.0003 --batch-size 40 --arch resnet50 \
--data-name ucf101 --representation iframe \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_split3_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_split3_test.txt \
--model-prefix kinetics_pretrained_ucf101_iframe_rolled_model_resnet50_seed_${seed}_split3_v3 \
--lr-steps 150 270 390 --epochs 510 --split split3 \
--gpus 0 1 --random_seed $seed --optimizer adam
done