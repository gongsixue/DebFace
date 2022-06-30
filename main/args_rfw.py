import os

# ===================== Visualization Settings =============================
port = 8096
env = 'main'
same_env = True
# ===================== Visualization Settings =============================

# ======================== Main Setings ====================================
log_type = 'traditional'
train = 'rfw'
save_results = False
result_path = '/research/prip-gongsixu/codes/biasface/results/models/debface_rfw2'
extract_feat = False
just_test = True
feat_savepath = '/research/prip-gongsixu/codes/biasface/results/features/feat_debface_biaslist_eccv2020.npz'
# resume = None
# resume = '/research/prip-gongsixu/results/models/teacher/sphere20a_20171020.pth'
# resume = '/research/prip-gongsixu/results/models/teacher/facenet_vgg2.pth'
# resume = '/scratch/gongsixue/face_resolution/models/model_ir_se50.pth'
resume = '/research/prip-gongsixu/codes/biasface/results/models/debface_rfw2/Save'
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
# dataset_root_test = '/research/prip-gongsixu/datasets/IJBA/IJBA_AlignAsSphereface'
# dataset_root_test = '/user/pripshare/Databases/FaceDatabasesPublic'
# dataset_root_test = '/research/prip-gongsixu/datasets/LFW/lfw_aligned_retina_112'
dataset_root_test = None

# dataset_root_train = '/research/prip-shiyichu/repo/insightface/datasets/faces_emore_images'
dataset_root_train = '/research/prip-gongsixu/datasets/RFW'

# preprocessing
preprocess_train = {"Resize": True, 
    "CenterCrop": True,
    # "RandomCrop": "True",
    "RandomHorizontalFlip": True, 
    # "RandomVerticalFlip": True,  
    # "RandomRotation": 10,
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}

preprocess_test = {"Resize": True, 
    "CenterCrop": True, 
    # "RandomCrop": True, 
    # "RandomHorizontalFlip": True, 
    # "RandomVerticalFlip": True, 
    # "RandomRotation": 10, 
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}

loader_input = 'loader_image'
loader_label = 'loader_numpy'

dataset_train = 'CSVListLoader'
# dataset_train = 'ClassSamplesDataLoader'
# dataset_train = 'FileListLoader'
# dataset_train = 'H5pyCSVLoader'
input_filename_train = '/research/prip-gongsixu/datasets/RFW/attr_rfw_balance_aligned_112.txt'
# input_filename_train = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
#     '/research/prip-gongsixu/codes/biasface/datasets/list_msceleb_demog.csv']
label_filename_train = None
dataset_options_train = {'ifile':input_filename_train, 'root':dataset_root_train,
                 'transform':preprocess_train, 'loader':loader_input}
# dataset_options_train = {'root':dataset_root_train, 'ifile':input_filename_train,
#                  'num_images':10, 'transform':preprocess_test, 'loader':loader_input,\
#                  'train_type':train}

dataset_test = 'CSVListLoader'
# dataset_test = 'AgeBinaryLoader'
# dataset_test = 'FileListLoader'
# input_filename_test = '/research/prip-gongsixu/datasets/LFW/list_lfw_aligned_retina_112.txt'
# input_filename_test = '/research/prip-gongsixu/datasets/RFW/attr_rfw_test_African_aligned_112.txt'
# input_filename_test = '../datasets/list_face_demog_groups_eccv2020.csv'
input_filename_test = '/research/prip-gongsixu/datasets/RFW/attr_rfw_test_aligned_112.txt'
label_filename_test = None
dataset_options_test = {'ifile':input_filename_test, 'root':dataset_root_test,
                 'transform':preprocess_test, 'loader':loader_input}

save_dir = os.path.join(result_path,'Save')
logs_dir = os.path.join(result_path,'Logs')
# ======================= Data Setings =====================================

# ======================= Network Model Setings ============================
# cpu/gpu settings
cuda = True
ngpu = 1
nthreads = 1

# nclasses = 85742
nclasses = 28000 # balance

# model_type = 'resnet18'
# model_options = {"nchannels":3,"nfeatures":512}
# model_type = 'incep_resnetV1'
# model_options = {"classnum": 10575, "features":False}

# model_type = 'sphereface20'
# model_type = 'sphereage10'
# model_type = 'rgsage4'
# model_options = {"nchannels":3, "nfilters":64, \
#     "ndim":512, "nclasses":nclasses, "dropout_prob":0.4, "features":False}
model_type = ['resnet_face50', 'FCnet']
model_options = [{"nfeatures":2*512, "nclasses": nclasses}, {"in_dims":2*512, "out_dims":2}]

# model_type = 'Backbone'
# model_options = {"num_layers": 50, "drop_ratio": 0.6, "mode": 'ir_se'}

# loss_type = 'Classification'
# loss_options = {"if_cuda":cuda}
# loss_type = 'BinaryClassify'
# loss_options = {"weight_file":'/research/prip-gongsixu/codes/biasface/datasets/weights_binary_classifier.npy',\
#     "if_cuda":cuda}
# loss_type = 'Regression'
# loss_options = {}
loss_type = ['AM_Softmax', 'Softmax', 'CrossEntropy', 'Classification']
# loss_options = {"nfeatures":512, "nclasses":nclasses, "m":20.0, "if_cuda":cuda}
loss_options = [{"nfeatures":512, "nclasses":nclasses, "s":64.0, "m":0.35},\
    {"nfeatures":512, "nclasses":4, "if_cuda":cuda},\
    {"if_cuda":cuda}, {"if_cuda":cuda}]

# input data size
input_high = 112
input_wide = 112
resolution_high = 112
resolution_wide = 112
# ======================= Network Model Setings ============================

# ======================= Training Settings ================================
# initialization
manual_seed = 0
nepochs = 300
epoch_number = 0

# batch
batch_size = 256
test_batch_size = 110

# optimization
# optim_method = 'Adam'
# optim_options = {"betas": (0.9, 0.999)}
optim_method = "SGD"
optim_options = {"momentum": 0.9, "weight_decay": 5e-4}

# learning rate
learning_rate = 1e-1
# scheduler_method = 'CosineAnnealingLR'
scheduler_method = 'Customer'
scheduler_options = {"T_max": nepochs, "eta_min": 1e-6}
lr_schedule = [15, 30, 50]
# lr_schedule = [8, 13, 15]
# lr_schedule = [45000,120000,195000,270000]
# ======================= Training Settings ================================

# ======================= Evaluation Settings ==============================
# label_filename = os.path.join('/research/prip-gongsixu/results/feats/evaluation', 'list_lfwblufr.txt')
label_filename = input_filename_test

# protocol and metric
protocol = 'RFW'
metric = 'cosine'

# files related to protocols
# IJB
eval_dir = '/research/prip-gongsixu/results/evaluation/ijbb/sphere/cs3'
# eval_dir = '/research/prip-gongsixu/results/evaluation/ijba'
imppair_filename = os.path.join(eval_dir, 'imp_pairs.csv')
genpair_filename = os.path.join(eval_dir, 'gen_pairs.csv')
pair_index_filename={'imposter':imppair_filename,'genuine':genpair_filename}
# pair_index_filename = eval_dir
template_filename = os.path.join(eval_dir, 'temp_dict.pkl')

# LFW
pairs_filename = '/research/prip-gongsixu/results/evaluation/lfw/lfw_pairs.txt'
nfolds=10

# RFW
# pairs_filename = '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt'
pairs_filename = {'African': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/African/African_pairs.txt',\
    'Asian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Asian/Asian_pairs.txt',\
    'Caucasian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Caucasian/Caucasian_pairs.txt',\
    'Indian': '/user/pripshare/Databases/FaceDatabases/RFW/test/txts/Indian/Indian_pairs.txt'}

# features saved as npm
nimgs=None
ndim=None

evaluation_type = 'FaceVerification'
evaluation_options = {'label_filename': label_filename,\
    'protocol': protocol, 'metric': metric,\
    'nthreads': nthreads, 'multiprocess':True,\
    'pair_index_filename': pair_index_filename,'template_filename': template_filename,\
    'pairs_filename': pairs_filename, 'nfolds': nfolds,\
    'nimgs': nimgs, 'ndim': ndim}
# evaluation_type = 'Top1Classification'
# evaluation_type = 'BiOrdinalClassify'
# evaluation_type = 'Agergs_classification'
# evaluation_options = {}
# ======================= Evaluation Settings ==============================

'''
epoch 33
Accuracy of African is 0.932
Accuracy of Asian is 0.9433333333333334
Accuracy of Caucasian is 0.9585
Accuracy of Indian is 0.949
Test [1/300]] ACC 0.9592

epoch 32
Accuracy of African is 0.933
Accuracy of Asian is 0.9431666666666667
Accuracy of Caucasian is 0.9591666666666666
Accuracy of Indian is 0.9485
Test [1/300]] ACC 0.9442

epoch 31
Accuracy of African is 0.9366666666666666
Accuracy of Asian is 0.9433333333333334
Accuracy of Caucasian is 0.9595
Accuracy of Indian is 0.9478333333333333
Test [1/300]] ACC 0.8322

epoch 30
Accuracy of African is 0.9358333333333333
Accuracy of Asian is 0.9428333333333333
Accuracy of Caucasian is 0.9618333333333333
Accuracy of Indian is 0.9483333333333334
Test [1/300]] ACC 0.9535

epoch 29
Accuracy of African is 0.921
Accuracy of Asian is 0.9238333333333333
Accuracy of Caucasian is 0.948
Accuracy of Indian is 0.9365
Test [1/300]] ACC 1.0765
'''