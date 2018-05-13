import cv2
import os
import numpy
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def extract_features(data_files, feat_detect, bf, ref_descs):
    """
    creating list of features extracted from a folder
    :param feat_detect: feature detector to use
    :param bf: brute_force matcher
    :param ref_descs: reference descriptor
    :return: list of all descriptors
    """
    train_descs = list()
    all_descs = list()

    for file in data_files:
        img = cv2.imread('image/train/' + file)
        kps, desc = get_features(img, feat_detect)
        matches = get_match(bf, desc, ref_descs)
        m_desc = list()
        for match in matches:
            m_desc.append(desc[match.queryIdx])
        train_descs.append(m_desc)
        all_descs.extend(m_desc)
    return train_descs, all_descs


def get_features(image, feature_detector):
    """
    extract features from image given a feature detector
    :param image: given image
    :param feature_detector: feature detector to use
    :return: list of keypoint and feature
    """
    gs_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, descriptors = feature_detector.detectAndCompute(gs_image, mask=None)
    if descriptors is None:
        return kp, None
    return kp, numpy.array(descriptors)


def bag_of_features(k_means, descs, hist_length):
    """
    creating bag of features
    :param k_means: k means classifier
    :param descs: descriptor list
    :param hist_length: histogram length
    :return: return bag of word
    """
    img_hist = [0] * hist_length
    for desc in descs:
        img_hist[k_means.predict([desc])[0]] += 1
    return img_hist


def initializing_classifier(clust_cnt):
    """
    initializing k-means and classifier
    :param clust_cnt: # of cluster
    :return: classifiers
    """
    classifier = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='brute')
    kmeans = KMeans(clust_cnt)
    return classifier, kmeans


def get_match(bf, descs, ref_desc):
    """
    matches descriptor with reference descriptor using brute_force matcher
    :param bf: brute_force matcher
    :param descs: descriptor to match
    :param ref_desc: descriptor to match with
    :return: return list of match descriptors
    """
    matches = bf.match(descs, ref_desc)
    dist = [m.distance for m in matches]
    dist_thres = (sum(dist) / len(dist))
    matches = [m for m in matches if m.distance < dist_thres]
    return matches


clust_cnt = 100
feat_detect = cv2.ORB_create()

# doing brute force match
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
ref_pic = cv2.imread("image/temoc.jpg")
ref_kps, ref_descs = get_features(ref_pic, feat_detect)
data_files = os.listdir('image/train')
print('extracting features ........')

train_descs, all_descs = extract_features(data_files, feat_detect, bf, ref_descs)

classifier, kmeans = initializing_classifier(clust_cnt)
print('clustering descriptors ........')

kmeans.fit(all_descs)
train_data = list()
classes = [int(file[0]) for file in data_files]

for img_idx in range(len(train_descs)):
    img_descriptors = train_descs[img_idx]
    file_name = data_files[img_idx]
    img_hist = bag_of_features(kmeans, img_descriptors, clust_cnt)
    train_data.append(img_hist)

print('training classifier ........')
classifier.fit(train_data, classes)

print('opening camera')
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:
        break
    else:
        feat, desc = get_features(frame, feat_detect)
        if not (desc is None):
            matches = get_match(bf, desc, ref_descs)
            m_desc = list()
            m_feat = list()
            for m in matches:
                m_feat.append(feat[m.queryIdx].pt)
                m_desc.append(desc[m.queryIdx])

            if not (m_desc is None):
                img_hist = bag_of_features(kmeans, m_desc, clust_cnt)
                result = classifier.predict([img_hist])[0]
                xmin = int(min(m_feat, key=lambda x: x[0])[0])
                xmax = int(max(m_feat, key=lambda x: x[0])[0])
                ymin = int(min(m_feat, key=lambda x: x[1])[1])
                ymax = int(max(m_feat, key=lambda x: x[1])[1])
                if result == 1:
                    cv2.putText(frame, "TEMOC", (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 102, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (127, 127, 127), thickness=4)
                else:
                    cv2.putText(frame, "not TEMOC", (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2, cv2.LINE_4)
    cv2.imshow("Live Image", frame)
    cnt = 5
    while cnt > 0:
        camera.read()
        cnt -= 1
cv2.destroyAllWindows()
camera.release()
