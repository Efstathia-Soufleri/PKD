data_root='/path-to-dataset'
for dataset in "ucf101"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python train_team_ce.py --lr 0.02 --is_train --split $split --random_seed $seed --batch-size 128 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 5 10 15 --epochs 30 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 12 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_ce_ic.txt"
            echo "python train_prog_kd.py --lr 0.01 --is_train --split $split --random_seed $seed --batch-size 64 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 25 30 35 --epochs 40 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 12 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_prog_kd_ic_64.txt"
        done
    done
done

for dataset in "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python train_team_ce.py --lr 0.02 --is_train --split $split --random_seed $seed --batch-size 128 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 5 10 15 --epochs 30 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 12 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_ce_ic.txt"
            echo "python train_prog_kd.py --lr 0.02 --is_train --split $split --random_seed $seed --batch-size 64 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 25 30 35 --epochs 40 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 12 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_prog_kd_ic_64.txt"
        done
    done
done

for dataset in "ucf101" "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            for ic in "IC1" "IC2" "IC3"; do
                echo "python save_logits.py --ic $ic --split $split --seed $seed --batch-size 128 --data-name ${dataset} --workers 8 --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt | tee ./logs/save_logits/save_${dataset}_kinetics_pretrained_${split}_${seed}_prog_kd_${ic}.txt"
            done
        done
    done
done        

for dataset in "ucf101" "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            for ic in "IC2" "IC3"; do
                echo "python wise_team_net_logit.py --ic $ic --split $split --seed $seed --batch-size 128 --data-name ${dataset}  | tee ./logs/wise/wise_logits_${dataset}_kinetics_pretrained_${split}_${seed}_prog_kd_${ic}.txt"
            done
        done
    done
done        

for dataset in "ucf101"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python train_i_frame.py --lr 0.01 --is_train --split $split --random_seed $seed --batch-size 64 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 15 20 25 --epochs 30 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 8 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_i_frame_ic.txt"
        done
    done
done

for dataset in "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python train_i_frame.py --lr 0.02 --is_train --split $split --random_seed $seed --batch-size 64 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 18 24 28 --epochs 34 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 8 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_i_frame_ic.txt"
        done
    done
done

for dataset in "ucf101" "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python train_prog_kd_anti.py --lr 0.01 --is_train --split $split --random_seed $seed --batch-size 64 --arch resnet50 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt --lr-steps 15 20 25 --epochs 30 --num_segments 8 --dropout 0.5 --wd 1e-4 --eval-freq 1 --gpus 0 --workers 8 | tee ./logs/${dataset}_kinetics_pretrained_${split}_${seed}_anti_prog_kd_64.txt"
        done
    done
done

for dataset in "ucf101" "hmdb51"; do
    for seed in 1 736 4711; do
        for split in "split1" "split2" "split3"; do
            echo "python team_net_wise_all_acc.py --split $split --seed $seed --batch-size 128 --data-name ${dataset} --data-root ${data_root}/${dataset} --train-list ${data_root}/${dataset}/${dataset}_${split}_train.txt --test-list ${data_root}/${dataset}/${dataset}_${split}_test.txt | tee ./logs/wise/wise_${dataset}_kinetics_pretrained_${split}_${seed}_prog_kd_all.txt"
        done
    done
done  