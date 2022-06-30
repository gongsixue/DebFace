import os

# ===================== Visualization Settings =============================
port = 8095
env = 'main'
same_env = True
# ===================== Visualization Settings =============================

# ======================== Main Setings ====================================
log_type = 'traditional'
train = 'face_cls'
save_results = True
result_path = '/research/prip-gongsixu/codes/biasface/results/models/face_demog'
extract_feat = False
just_test = True
feat_savepath = '/research/prip-gongsixu/codes/biasface/results/features/feat_debface_subfig.npz'
# resume = None
# resume = '/research/prip-gongsixu/results/models/teacher/sphere20a_20171020.pth'
# resume = '/research/prip-gongsixu/results/models/teacher/facenet_vgg2.pth'
# resume = '/scratch/gongsixue/face_resolution/models/model_ir_se50.pth'
resume = '/research/prip-gongsixu/codes/biasface/results/models/debface/Save'
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
# dataset_root_test = '/research/prip-gongsixu/datasets/IJBA/IJBA_AlignAsSphereface'
dataset_root_test = '/user/pripshare/Databases/FaceDatabasesPublic'
# dataset_root_test = '/research/prip-gongsixu/datasets/LFW/lfw_aligned_retina_112'
dataset_root_train = '/research/prip-shiyichu/repo/insightface/datasets/faces_emore_images'
# dataset_root_train = '/scratch/gongsixue/datasets/LFW/LFW_AlignAsSphereface'

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

# dataset_train = 'AFAD'
# dataset_train = 'AgeCSVListLoader'
# dataset_train = 'AgeBinaryLoader'
# dataset_train = 'ClassSamplesDataLoader'
# input_filename_train = '/research/prip-gongsixu/results/feats/evaluation/list_lfwblufr.txt'
# input_filename_train = ['/scratch/gongsixue/msceleb_AlignedAsArcface_images.hdf5',\
#     '/research/prip-gongsixu/codes/biasface/datasets/list_msceleb_demog.csv']

input_filename_train = 'datasets/AFAD/afad_train.csv'
label_filename_train = None
# dataset_train = 'FileListLoader'
# dataset_train = 'H5pyCSVLoader'
dataset_train = 'GenderCSVListLoader'
dataset_options_train = {'ifile':input_filename_train, 'root':dataset_root_train,
                 'transform':preprocess_train, 'loader':loader_input}
# dataset_options_train = {'root':dataset_root_train, 'ifile':input_filename_train,
#                  'num_images':10, 'transform':preprocess_test, 'loader':loader_input,\
#                  'train_type':train}

dataset_test = 'GenderCSVListLoader'
# dataset_test = 'AgeBinaryLoader'
# dataset_test = 'AFAD'
# dataset_test = 'FileListLoader'
# input_filename_test = '/research/prip-gongsixu/datasets/LFW/list_lfw_aligned_retina_112.txt'
# input_filename_test = '/research/prip-gongsixu/datasets/IJBA/list_ijba_aligned.txt'
input_filename_test = 'datasets/AFAD/afad_test.csv'
# input_filename_test = '../datasets/list_face_demog_groups_eccv2020.csv'
# input_filename_test = '/user/pripshare/Databases/FaceDatabasesPublic/CACD/CACDVS_AlignAsSphereface/cacdvs_aligned.txt'
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

nclasses = 85742

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
model_options = [{"nfeatures": 4*512, "nclasses": nclasses}, {"in_dims":4*512, "out_dims":2}]

# model_type = 'Backbone'
# model_options = {"num_layers": 50, "drop_ratio": 0.6, "mode": 'ir_se'}

# loss_type = 'Classification'
# loss_options = {"if_cuda":cuda}
# loss_type = 'BinaryClassify'
# loss_options = {"weight_file":'/research/prip-gongsixu/codes/biasface/datasets/weights_binary_classifier.npy',\
#     "if_cuda":cuda}
# loss_type = 'Regression'
# loss_options = {}
loss_type = ['AM_Softmax', 'Softmax', 'Softmax', 'Softmax', 'CrossEntropy', 'Classification']
# loss_options = {"nfeatures":512, "nclasses":nclasses, "m":20.0, "if_cuda":cuda}
loss_options = [{"nfeatures":512, "nclasses":nclasses, "s":64.0, "m":0.35},\
    {"nfeatures":512, "nclasses":2, "if_cuda":cuda}, {"nfeatures":512, "nclasses":6, "if_cuda":cuda}, \
    {"nfeatures":512, "nclasses":4, "if_cuda":cuda}, \
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
test_batch_size = 50

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
protocol = 'BLUFR'
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
