voc_id_name_map = {
	1: 'aeroplane',2: 'bicycle',3: 'bird',4: 'boat',5: 'bottle',6: 'bus',
	7: 'car',8: 'cat',9: 'chair',10: 'cow',11: 'diningtable',12: 'dog',
	13: 'horse',14: 'motorbike',15: 'person',16: 'pottedplant',17: 'sheep',
	18: 'sofa',19: 'train',20: 'tvmonitor',
}

ret = []

for i,j in voc_id_name_map.items():
    if i < 11:
        ret.append(j)

print(ret)