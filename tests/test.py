
fp_label_txt = './test-object-detection-yolov8/classes.txt'
with open(fp_label_txt, 'r') as f:
    labels = f.readlines()
    labels = [l.strip() for l in labels]
print(labels)