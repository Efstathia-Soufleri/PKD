#MV
for split in 'split1' 'split2' 'split3'
do
for representation in 'mv'
do
for seed in 4711 1 736
do
python train_freeze_backbone_IC_KD_iframe.py --lr 0.01 --batch-size 32 --arch resnet18 \
--data-name ucf101 --representation $representation \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_${split}_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
--model-prefix ucf101_model_IC_kinetics_pretrained_${representation}_KD_iframe_seed_${seed}_${split} \
--lr-steps 50 100 --epochs 150 --split $split --optimizer adam \
--gpus 0 1 2 3 --random_seed $seed | tee ./logs/ucf101_kinetics_pretrained/KD_one_teacher/${representation}/KD_iframe_IC_${representation}_seed_${seed}_${split}.txt
done
done
done


#RESIDUAL
for split in 'split1' 'split2' 'split3'
do
for representation in 'residual'
do
for seed in 4711 1 736
do
python train_freeze_backbone_IC_KD_iframe.py --lr 0.005 --batch-size 32 --arch resnet18 \
--data-name ucf101 --representation $representation \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_${split}_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
--model-prefix ucf101_model_IC_kinetics_pretrained_${representation}_KD_iframe_seed_${seed}_${split} \
--lr-steps 50 100 --epochs 150 --split $split --optimizer adam \
--gpus 0 1 2 3 --random_seed $seed | tee ./logs/ucf101_kinetics_pretrained/KD_one_teacher/${representation}/KD_iframe_IC_${representation}_seed_${seed}_${split}.txt
done
done
done


#IFRAME
for split in 'split1' 'split2' 'split3'
do
for representation in 'iframe'
do
for seed in 4711 1 736
do
python train_freeze_backbone_IC_KD_iframe.py --lr 0.0003 --batch-size 32 --arch resnet50 \
--data-name ucf101 --representation $representation \
--data-root /home/path-to-dataset/ucf101/mpeg4_videos \
--train-list /home/path-to-dataset/datalists/ucf101_${split}_train.txt \
--test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
--model-prefix ucf101_model_IC_kinetics_pretrained_${representation}_KD_iframe_seed_${seed}_${split} \
--lr-steps 50 100 --epochs 150 --split $split --optimizer adam \
--gpus 0 1 2 3 --random_seed $seed | tee ./logs/ucf101_kinetics_pretrained/KD_one_teacher/${representation}/KD_iframe_IC_${representation}_seed_${seed}_${split}.txt
done
done
done

