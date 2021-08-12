# standard

python mytrain.py DATASETS.TRAIN "('voc[1,10]_train', )" OUTPUT_DIR './output/voc[1,10]'
python mytrain.py DATASETS.TRAIN "('voc[1,15]_train', )" OUTPUT_DIR './output/voc[1,15]'
python mytrain.py DATASETS.TRAIN "('voc[1,19]_train', )" OUTPUT_DIR './output/voc[1,20]'

python mytrain.py --config-file "myILOD/configs/base.yaml" DATASETS.TRAIN "('voc[11,20]_train', )" OUTPUT_DIR './output/voc[11,20]'
python mytrain.py --config-file "myILOD/configs/base.yaml" DATASETS.TRAIN "('voc[16,20]_train', )" OUTPUT_DIR './output/voc[16,20]'
python mytrain.py --config-file "myILOD/configs/base.yaml" DATASETS.TRAIN "('voc[20,20]_train', )" OUTPUT_DIR './output/voc[20,20]'

# freeze



# pb body

python mytrain.py --config-file "myILOD/configs/pb[body].yaml" DATASETS.TRAIN "('voc[1,10]_train', )" OUTPUT_DIR './output/pb[body]voc[1,10]'
python mytrain.py --config-file "myILOD/configs/pb[body].yaml" DATASETS.TRAIN "('voc[1,15]_train', )" OUTPUT_DIR './output/pb[body]voc[1,15]'
python mytrain.py --config-file "myILOD/configs/pb[body].yaml" DATASETS.TRAIN "('voc[1,19]_train', )" OUTPUT_DIR './output/pb[body]voc[1,19]'

python mytrain.py --config-file "myILOD/configs/pb[body].yaml" DATASETS.TRAIN "('voc[11,20]_train', )" OUTPUT_DIR './output/pb[body]voc[11,20]'
python mytrain.py --config-file "myILOD/configs/pb[body].yaml" DATASETS.TRAIN "('voc[16,20]_train', )" OUTPUT_DIR './output/pb[body]voc[16,20]'
python mytrain.py --config-file "myILOD/configs/pb[body]+1.yaml" DATASETS.TRAIN "('voc[20,20]_train', )" OUTPUT_DIR './output/pb[body]voc[20,20]'

# pb roi_heads

python mytrain.py --config-file "myILOD/configs/pb[roi_heads]+1.yaml" DATASETS.TRAIN "('voc[20,20]_train', )" MODEL.WEIGHTS "./output/voc[1,19]/model_final.pth" OUTPUT_DIR './output/pb[:roi]19+1'
python mytrain.py --config-file "myILOD/configs/pb[roi_heads].yaml" DATASETS.TRAIN "('voc[16,20]_train', )" MODEL.WEIGHTS "./output/voc[1,15]/model_final.pth" OUTPUT_DIR './output/pb[:roi]15+5'
python mytrain.py --config-file "myILOD/configs/pb[roi_heads].yaml" DATASETS.TRAIN "('voc[11,20]_train', )" MODEL.WEIGHTS "./output/voc[1,10]/model_final.pth" OUTPUT_DIR './output/pb[:roi]10+10'

# pb rpn

python mytrain.py --config-file "myILOD/configs/pb[rpn]+1.yaml" DATASETS.TRAIN "('voc[20,20]_train', )" MODEL.WEIGHTS "./output/voc[1,19]/model_final.pth" OUTPUT_DIR './output/pb[:rpn]19+1'
python mytrain.py --config-file "myILOD/configs/pb[rpn].yaml" DATASETS.TRAIN "('voc[16,20]_train', )" MODEL.WEIGHTS "./output/voc[1,15]/model_final.pth" OUTPUT_DIR './output/pb[:rpn]15+5'
python mytrain.py --config-file "myILOD/configs/pb[rpn].yaml" DATASETS.TRAIN "('voc[11,20]_train', )" MODEL.WEIGHTS "./output/voc[1,10]/model_final.pth" OUTPUT_DIR './output/pb[:rpn]10+10'
