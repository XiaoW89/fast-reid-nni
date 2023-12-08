from nni.experiment import Experiment

experiment = Experiment('local')
experiment.name = "reid-sbs-elan"
search_space = {
    'LR': {'_type': 'loguniform', '_value': [0.0005, 0.05]},
    'WD': {'_type': 'loguniform', '_value': [5e-4, 5e-3, 5e-2, 0.5]},
    'PoolType':{'_type': 'choice', '_value': ['GlobalAvgPool', 'GlobalMaxPool', 'AdaptiveAvgMaxPool']}, 
    'BatchSize':{'_type': 'choice', '_value': [128, 256, 512, 1024]}, 
    #'MaxEpoch':{'_type': 'choice', '_value': [201, 301, 401, 501]}, 
    'MaxEpoch':{'_type': 'choice', '_value': [10, 20]}, 
    'Optimizer': {'_type': 'choice', '_value': ['Adam', 'AdamW']},
    'HeadMargin':{'_type': 'loguniform', '_value': [0.1, 2.0]}, 
    'HeadScale':{'_type': 'choice', '_value': [16, 32, 64, 128, 256]}, 
}
experiment.config.trial_command = 'python3 tools/train_net.py --config-file ./configs/Market1501/sbs_elan_aug.yml --nni-hpo --num-gpus 1 MODEL.BACKBONE.WITH_IBN False OUTPUT_DIR logs/market1501/sbs_elan_96x32_22222_cropbody_nni_v1 TEST.EVAL_PERIOD 40'

experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 3
experiment.config.trial_gpu_number = 1 
experiment.config.training_service.use_active_gpu = False

experiment.run(8081)
experiment.stop()
