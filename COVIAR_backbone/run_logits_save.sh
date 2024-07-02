# SAVE THE LOGITS FOR UCF-101

# for all splits
for split in 'split1' 'split2' 'split3'
do
for representation in 'iframe'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet50 \
    --data-name ucf101 --representation iframe \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/ucf101_model_IC_kinetics_pretrained_iframe_KD_progressive_seed_4711_${split}_iframe_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet50 \
    --data-name ucf101 --representation iframe \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_ucf101_iframe_rolled_model_resnet50_seed_4711_${split}_v3_iframe_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done


#MV
# for all splits
for split in 'split1' 'split2' 'split3'
do
for representation in 'mv'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet18 \
    --data-name ucf101 --representation $representation \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/ucf101_model_IC_kinetics_pretrained_${representation}_KD_progressive_seed_4711_${split}_${representation}_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet18 \
    --data-name ucf101 --representation $representation \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_${representation}_ucf101_${representation}_rolled_model_resnet18_seed_4711_${split}_v2_${representation}_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done


for split in 'split1' 'split2' 'split3'
do
for representation in 'residual'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet18 \
    --data-name ucf101 --representation $representation \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/ucf101_model_IC_kinetics_pretrained_${representation}_KD_progressive_seed_4711_${split}_${representation}_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet18 \
    --data-name ucf101 --representation $representation \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_${representation}_ucf101_${representation}_rolled_model_resnet18_seed_4711_${split}_v2_${representation}_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done

# SAVE THE LOGITS FOR HMDB-51

# for all splits
for split in 'split1' 'split2' 'split3'
do
for representation in 'iframe'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet50 \
    --data-name hmdb51 --representation iframe \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/hmdb_model_IC_iframe_KD_progressive_seed_4711_${split}_300_epochs_v2_IC1_IC2_modified_iframe_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet50 \
    --data-name hmdb51 --representation iframe \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_hmdb51_iframe_rolled_model_resnet50_seed_4711_${split}_v3_iframe_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done


#MV
# for all splits
for split in 'split1' 'split2' 'split3'
do
for representation in 'mv'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet18 \
    --data-name hmdb51 --representation $representation \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/hmdb_model_IC_${representation}_KD_progressive_seed_4711_${split}_${representation}_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet18 \
    --data-name hmdb51 --representation $representation \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_${representation}_hmdb51_${representation}_rolled_model_resnet18_seed_4711_${split}_v2_${representation}_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done


for split in 'split1' 'split2' 'split3'
do
for representation in 'residual'
do
for test_segments in 1 2 3 4 5 6 7 8 9 10 11 12 14 16 18 20 
do
for IC in 1 2 3
do
python test_save_logits_IC.py --gpus 0 \
    --arch resnet18 \
    --data-name hmdb51 --representation $representation \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/hmdb_model_IC_${representation}_KD_progressive_seed_4711_${split}_${representation}_checkpoint_KD_progressive_IC${IC}_seed_4711.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC $IC --split $split
done

python test_save_logits_IC4.py --gpus 0 \
    --arch resnet18 \
    --data-name hmdb51 --representation $representation \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --weights ./checkpoints/kinetics_pretrained_${representation}_hmdb51_${representation}_rolled_model_resnet18_seed_4711_${split}_v2_${representation}_model_best.pth.tar \
    --test_segments $test_segments --learning_type progressive_KD --IC 4 --split $split
done
done
done