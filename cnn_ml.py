# _*_coding:utf-8 _*_

import torch
from sklearn.svm import SVC

# from sklearn.externals import joblib
import joblib
import pandas as pd
import cfg
import os
from PIL import Image
from data.transform import get_test_transform
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np
# from The_Feature import FeatureExtractor
import warnings
warnings.filterwarnings("ignore")

# torch.nn.Module.dump_patches = True
extract_list = ["layer4"]
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    # 自己修改forward函数
    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            # if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
                # print(x)
                break
        return outputs

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    # print(checkpoint)
    # model = checkpoint['model']  # 提取网络结构
    model = torch.load("./data/weights/resnet50/model.pth", map_location=torch.device('cpu'))
    # print(model)
    # model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


NB_features = 2048     # resnet50此处为2048 resnet18此处为512,看自己使用什么网络
def save_feature(model, feature_path, label_path):
    '''
    提取特征，保存为pkl文件
    '''
    model = load_checkpoint(model)
    # print(model)
    # print(model.layer4)
    # for name in model.state_dict().keys():
        # print(name)
    print('..... Finished loading model! ......')
    ##将模型放置在gpu上运行
    if torch.cuda.is_available():
        model.cuda()
    ## 特征的维度需要自己根据特定的模型调整
    nb_features = NB_features
    features = np.empty((len(imgs), nb_features))
    labels = []
    print(range(len(imgs)))
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split(' ')[0]
        label = imgs[i].strip().split(' ')[1]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        # print(type(img))
        img = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            # out = model.extract_features(img)
            extract_result = FeatureExtractor(model, extract_list)
            out = extract_result(img)[0]
            # print(out.size())
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
            # print(out3.size())
            # print(out2.size())
        features[i, :] = feature.cpu().numpy()
        labels.append(label)

    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')



def classifier_training(feature_path, label_path, save_path):
    '''
    训练分类器
    '''
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open(feature_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))
    classifier = SVC(C=0.5)
    # classifier = MLPClassifier()
    # classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    # classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    # classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
    #                        max_features=50, max_depth=40, min_samples_leaf=4)
    # classifier = GaussianNB()
    print(".... Start fitting this classifier ....")
    classifier.fit(features, labels)
    print("... Training process is down. Save the model ....")
    joblib.dump(classifier, save_path)
    print("... model is saved ...")



def classifier_pred(model_path, feature, id):
    '''
    得到测试集的预测结果
    '''
    features = pickle.load(open(feature, 'rb'))
    ids = pickle.load(open(id, 'rb'))
    print("... loading the model ...")
    classifier = joblib.load(model_path)
    print("... load model done and start predicting ...")
    predict = classifier.predict(features)
    # print(type(predict))
    # print(predict.shape)
    # print(ids)
    prediction = predict.tolist()
    submission = pd.DataFrame({"ID": ids, "Label": prediction})
    submission.to_csv('../' + 'svm_submission.csv', index=False, header=False)




if __name__ == "__main__":
    # #构建保存特征的文件夹
    feature_path = '../features/'
    os.makedirs(feature_path, exist_ok=True)

#############################################################
    #### 保存训练集特征
    with open(cfg.TRAIN_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    train_feature_path = feature_path + 'psdufeature.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, train_feature_path, train_label_path)
    #
    ## #训练并保存分类器
    train_feature_path = feature_path + 'psdufeature.pkl'
    train_label_path = feature_path + 'psdulabel.pkl'
    save_path = feature_path + 'psdusvm.m'
    classifier_training(train_feature_path, train_label_path, save_path)

######################################################################
    # #保存测试集特征
    with open(cfg.VAL_LABEL_DIR, 'r')as f:
        imgs = f.readlines()
    test_feature_path = feature_path + 'testfeature.pkl'
    test_id_path = feature_path + 'testid.pkl'
    cnn_model = cfg.TRAINED_MODEL
    save_feature(cnn_model, test_feature_path, test_id_path)


    ## #预测结果
    test_feature_path = feature_path + 'testfeature.pkl'
    test_id_path = feature_path + 'testid.pkl'
    save_path = feature_path + 'psdusvm.m'
    classifier_pred(save_path, test_feature_path, test_id_path)
