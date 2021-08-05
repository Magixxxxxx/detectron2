python mytrain.py DATASETS.TRAIN "('voc[11,20]_train', )" OUTPUT_DIR './output/pb[body]voc[11,20]'
python mytrain.py DATASETS.TRAIN "('voc[16,20]_train', )" OUTPUT_DIR './output/pb[body]voc[16,20]'
python mytrain.py DATASETS.TRAIN "('voc[20,20]_train', )" OUTPUT_DIR './output/pb[body]voc[20,20]'

python mytrain.py DATASETS.TRAIN "('voc[1,10]_train', )" OUTPUT_DIR './output/voc[1,10]'
python mytrain.py DATASETS.TRAIN "('voc[1,15]_train', )" OUTPUT_DIR './output/voc[1,15]'
python mytrain.py DATASETS.TRAIN "('voc[1,19]_train', )" OUTPUT_DIR './output/voc[1,19]'
python mytrain.py DATASETS.TRAIN "('voc[1,20]_train', )" OUTPUT_DIR './output/voc[1,20]'

python mytrain.py --config-file "pb[roi_heads].yaml" 