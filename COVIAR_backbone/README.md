#### TRAIN THE BACKBONE NETWORKS FOR UCF-101 AND HMDB-51:

```
run_train_ucf101_kinetics_pretrained.sh
run_train_hmdb51_kinetics_pretrained.sh
```

#### TRAIN UCF-101 AND HMDB-51 WITH CE, Prog KD, Prog KD anticurriculum, Iframe KD:

```
run_training_CE.sh
run_training_KD.sh
run_training_KD_anticurriculum.sh
run_training_KD_iframe.sh
```

#### SAVE THE LOGITS FOR UCF-101 and HMDB-51 PRETRAINED ON KINETICS (MV,R,I)

```
run_logits_save.sh
```

#### SAVE THE LOGITS FOR UCF-101 and HMDB-51 PRETRAINED ON KINETICS (MV,R,I) FOR THE TRAIN SET

```
run_save_logits_trainset.sh
```

##### Get the numbers for the table for WISE and for the trade-off curve pretrained on kinetics:

##### UCF-101

```
for split in 'split1' 'split2' 'split3'
do
python wise_ucf101_kinetics_pretrained.py --gpus 0 \
    --arch resnet50 \
    --data-name ucf101 \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split}_test.txt \
    --test_segments_mv 1 --test_segments_r 1 --test_segments_i 1 --learning_type progressive_KD \
    --threshold 0.999999 --split $split
done
```


##### HMDB-51

```
for split in 'split1' 'split2' 'split3'
do
python hmdb_wise.py --gpus 0 \
    --arch resnet50 \
    --data-name hmdb51 \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split}_test.txt \
    --test_segments_mv 1 --test_segments_r 2 --test_segments_i 1 --learning_type progressive_KD \
    --threshold 0.999999 --split $split
done
```

#### PLOT THE TRADE-OFF CURVE FOR UCF-101 AND HMDB-51 PRETRAINED ON KINETICS:

```
python plot_trade_off_ucf101_kinetics_pretrained.py
```

```
python plot_acc_flops_trade_off_hmdb51.py
```

#### PLOT THE WISE TALBE FOR UCF-101 AND HMDB-51 (use split1,split2,split3):
```
python plot_wise_table.py
```

```
python plot_wise_table_hmdb51.py
```

#### PLOT THE ABALTION STUDY ON UCF-101 AND HMDB-51:

```
python plot_frame_ablation_study_ucf101.py
```

```
python plot_frame_ablation_study_hmdb51.py
```

#### COMPUTE # OF FRAMES ABLATION STUDY UCF-101 AND HMDB-51:

```
for split in 'split1' 'split2' 'split3'
do
python ablation_study_number_of_frames_ucf101.py --gpus 0 \
    --arch resnet50 \
    --data-name ucf101 \
    --data-root /home/path-to-dataset/ucf101/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/ucf101_${split_test}.txt \
    --test_segments_mv 1 --test_segments_r 2 --test_segments_i 1 --learning_type progressive_KD --split $split
done
```

```
for split in 'split1' 'split2' 'split3'
do
python ablation_study_number_of_frames_hmdb51.py --gpus 0 \
    --arch resnet50 \
    --data-name hmdb51 \
    --data-root /home/path-to-dataset/hmdb51/mpeg4_videos \
    --test-list /home/path-to-dataset/datalists/hmdb51_${split_test}.txt \
    --test_segments_mv 1 --test_segments_r 2 --test_segments_i 1 --learning_type progressive_KD --split $split
done
```

