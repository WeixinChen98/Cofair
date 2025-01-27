"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset, DiscriminatorDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader, DiscriminatorDataReader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_recommendation_model, get_fairness_model, dict2str, get_disc_trainer, get_trainer
import platform
import os

# FMMR
from collections import defaultdict
from torch.utils.data import DataLoader
from common.discriminators import BinaryDiscriminator, MulticlassDiscriminator
import torch

T = 5
ts = []
for t in range(0, T+1):
    ts.append(t)

# taus = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
taus = [0.0, 0.25, 0.5, 0.75, 1.0]


def quick_start(recommendation_model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(recommendation_model = recommendation_model, dataset = dataset, config_dict = config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split()
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    (valid_data, test_data) = (
        EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
        EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    


    # hyper-parameters
    hyper_ls = []
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))

        # create data reader
        disc_data_reader = DiscriminatorDataReader(path=config['data_path'], dataset_name=config['dataset'], feature_columns=config['feature_columns'], sep='\t', test_ratio=0.2)

        # set random state of dataloader
        train_data.pretrain_setup()

        # model loading and initialization
        fair_disc_dict = None
        if config['fairness_model'] is None:
            model = get_recommendation_model(config['recommendation_model'])(config, train_data).to(config['device'])
        else:
            model = get_fairness_model(config['fairness_model'])(config, train_data).to(config['device'])

            # create discriminators
            fair_disc_dict = {}
            for feat_idx in disc_data_reader.feature_info:
                if disc_data_reader.feature_info[feat_idx].num_class == 2:
                    fair_disc_dict[feat_idx] = BinaryDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                else:
                    fair_disc_dict[feat_idx] = MulticlassDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                fair_disc_dict[feat_idx].apply(fair_disc_dict[feat_idx].init_weights)

                if torch.cuda.device_count() > 0:
                    fair_disc_dict[feat_idx] = fair_disc_dict[feat_idx].cuda()

            
        logger.info(model)

        if config['model_load_path'] is not None:
            model.load_model(config['model_load_path'])

        # trainer loading and initialization
        trainer = get_trainer()(config, model, fair_disc_dict = fair_disc_dict)
        # debug
        # model training
        if config['fairness_model'] in ['DAL', 'DALS', 'DALF', 'DALFS', 'DALFD', 'DALFDI', 'DALFDT', 'SAF_D', 'SAF_D_SC', 'MF', 'DSPT1']:
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, ts = ts, valid_data=valid_data, test_data=test_data, saved=save_model)
        elif config['fairness_model'] in ['DALFDS']:
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, ts = taus, valid_data=valid_data, test_data=test_data, saved=save_model)
        else:
            best_valid_score, best_valid_result, best_test_upon_valid = trainer.fit(train_data, ts = taus, valid_data=valid_data, test_data=test_data, saved=save_model)
            # raise error('Double Check Here')

        # Train extra discriminator for evaluation
        # create data reader
        disc_data_reader = DiscriminatorDataReader(path=config['data_path'], dataset_name=config['dataset'], feature_columns=config['feature_columns'], sep='\t', test_ratio=0.2)

        # create data processor
        extra_data_processor_dict = {}
        for stage in ['train', 'test']:
            extra_data_processor_dict[stage] = DiscriminatorDataset(disc_data_reader, stage, batch_size = config['disc_batch_size'])




        model.load_model()
        model.freeze_model()
        disc_trainer = get_disc_trainer()(disc_epoch = config['disc_epoch'])




        if config['fairness_model'] in ['DAL', 'DALS', 'DALF', 'DALFS', 'DALFD', 'DALFDI', 'DALFDT', 'SAF_D', 'SAF_D_SC', 'MF', 'DSPT1']:
            for t in ts:
                logger.info('t = {}'.format(t))
                # create discriminators
                extra_fair_disc_dict = {}
                for feat_idx in disc_data_reader.feature_info:
                    if disc_data_reader.feature_info[feat_idx].num_class == 2:
                        extra_fair_disc_dict[feat_idx] = \
                            BinaryDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                    else:
                        extra_fair_disc_dict[feat_idx] = \
                            MulticlassDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                    if torch.cuda.device_count() > 0:
                        extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()
                    
                best_disc_upon_valid_user = disc_trainer.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, config['lr_attack'], config['l2_attack'], t = t)
        elif config['fairness_model'] in ['DALFDS']:
            for tau in taus:
                logger.info('tau = {}'.format(tau))
                # create discriminators
                extra_fair_disc_dict = {}
                for feat_idx in disc_data_reader.feature_info:
                    if disc_data_reader.feature_info[feat_idx].num_class == 2:
                        extra_fair_disc_dict[feat_idx] = \
                            BinaryDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                    else:
                        extra_fair_disc_dict[feat_idx] = \
                            MulticlassDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                    if torch.cuda.device_count() > 0:
                        extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()
                    
                best_disc_upon_valid_user = disc_trainer.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, config['lr_attack'], config['l2_attack'], t = tau)
        else:
            extra_fair_disc_dict = {}
            for feat_idx in disc_data_reader.feature_info:
                if disc_data_reader.feature_info[feat_idx].num_class == 2:
                    extra_fair_disc_dict[feat_idx] = \
                        BinaryDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                else:
                    extra_fair_disc_dict[feat_idx] = \
                        MulticlassDiscriminator(disc_data_reader.feature_info[feat_idx], embedding_size = config['embedding_size'], config = config)
                if torch.cuda.device_count() > 0:
                    extra_fair_disc_dict[feat_idx] = extra_fair_disc_dict[feat_idx].cuda()
                
            best_disc_upon_valid_user = disc_trainer.train_discriminator(model, extra_data_processor_dict, extra_fair_disc_dict, config['lr_attack'], config['l2_attack'])
        hyper_ret.append((hyper_tuple, best_valid_result, best_test_upon_valid, best_disc_upon_valid_user))



        # save best test
        if best_test_upon_valid[val_metric] > best_test_value:
            best_test_value = best_test_upon_valid[val_metric]
            best_test_idx = idx
            model.save_pretrained_representation()
        idx += 1


        logger.info('best valid result: {}'.format(dict2str(best_valid_result)))
        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('disc user result: {}'.format(dict2str(best_disc_upon_valid_user)))


        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Valid: {},\nTest: {},\nDisc_user: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1]), dict2str(hyper_ret[best_test_idx][2]), dict2str(hyper_ret[best_test_idx][3])))
        

        
    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v, d_u) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {},\n best disc from user: {},\n'.format(config['hyper_parameters'],
                                                                                    p, dict2str(k), dict2str(v), dict2str(d_u)))


    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nValid: {},\nTest: {},\nDisc user: {}\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1]),
                                                                   dict2str(hyper_ret[best_test_idx][2]),
                                                                   dict2str(hyper_ret[best_test_idx][3])))


