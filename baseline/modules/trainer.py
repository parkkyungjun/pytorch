"""Trainer
"""

from modules.metrics import ConfMatrix
from modules.losses import *
from modules.utils import *
from modules.datasets import batch_transform
from modules.utils import generate_unsup_data, label_onehot
from time import time
import pandas as pd
from tqdm import tqdm

class Trainer():

    def __init__(self,
                 model,
                 ema,
                 data_loader,
                 optimizer,
                 device,
                 config,
                 logger, 
                 interval=100):
        
        self.model = model
        self.ema = ema
        self.data_loader = data_loader
        self.config =config
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.interval = interval
        self.mIoU = 0

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0 # Epoch loss mean
        self.score_dict = dict()  # metric score
        self.elapsed_time = 0
        

    def train(self, train_l_loader, train_u_loader):

        train_l_dataset = iter(train_l_loader)
        train_u_dataset = iter(train_u_loader)
        self.model.train()
        self.ema.model.train()

        train_epoch = len(train_l_loader)
        start_timestamp = time()
        l_conf_mat = ConfMatrix(self.data_loader.num_segments)

        for i in range(train_epoch):
            train_l_data, train_l_label = train_l_dataset.next()
            train_l_data, train_l_label = train_l_data.to(self.device), train_l_label.to(self.device)

            train_u_data = train_u_dataset.next()
            train_u_data = train_u_data.to(self.device)

            self.optimizer.zero_grad()

            # generate pseudo-labels
            with torch.no_grad():
                pred_u, _ = self.ema.model(train_u_data)
                pred_u_large_raw = F.interpolate(pred_u, size=train_u_data.shape[2:], mode='bilinear',
                                                 align_corners=True)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u_large_raw, dim=1), dim=1)

                # random scale images first
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                    self.data_loader.crop_size, self.data_loader.scale_size, apply_augmentation=False)

                # apply mixing strategy: cutout, cutmix or classmix
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=self.config['apply_aug'])

                # apply augmentation: color jitter + flip + gaussian blur
                train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                    batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                    self.data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

            # generate labelled and unlabelled data loss
            pred_l, rep_l = self.model(train_l_data)
            pred_l_large = F.interpolate(pred_l, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            pred_u, rep_u = self.model(train_u_aug_data)
            pred_u_large = F.interpolate(pred_u, size=train_l_label.shape[1:], mode='bilinear', align_corners=True)

            rep_all = torch.cat((rep_l, rep_u))
            pred_all = torch.cat((pred_l, pred_u))

            # supervised-learning loss
            sup_loss = compute_supervised_loss(pred_l_large, train_l_label)

            # unsupervised-learning loss
            unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits,
                                                   self.config['strong_threshold'])

            # apply regional contrastive loss
            if self.config['apply_reco']:
                with torch.no_grad():
                    train_u_aug_mask = train_u_aug_logits.ge(self.config['weak_threshold']).float()
                    mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
                    mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')

                    label_l = F.interpolate(label_onehot(train_l_label, self.data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_u = F.interpolate(label_onehot(train_u_aug_label, self.data_loader.num_segments),
                                            size=pred_all.shape[2:], mode='nearest')
                    label_all = torch.cat((label_l, label_u))

                    prob_l = torch.softmax(pred_l, dim=1)
                    prob_u = torch.softmax(pred_u, dim=1)
                    prob_all = torch.cat((prob_l, prob_u))

                reco_loss = compute_reco_loss(rep_all, label_all, mask_all, prob_all, self.config['strong_threshold'],
                                              self.config['temp'], self.config['num_queries'], self.config['num_negatives'])
            else:
                reco_loss = torch.tensor(0.0)

            loss = sup_loss + unsup_loss + reco_loss
            loss.backward()
            self.optimizer.step()
            self.ema.update(self.model)
            l_conf_mat.update(pred_l_large.argmax(1).flatten(), train_l_label.flatten())

            # History
            self.loss_sum += loss.item()

            # Logging
            if i % self.interval == 0:
                msg = f"batch: {i}/{train_epoch} loss: {loss.item()}"
                self.logger.info(msg)
                
        # Epoch history
        self.loss_mean = self.loss_sum / train_epoch  # Epoch loss mean
        self.mIoU, _ = l_conf_mat.get_metrics()
        self.score_dict['mIoU'] = self.mIoU

        # Elapsed time
        end_timestamp = time()
        self.elapsed_time = end_timestamp - start_timestamp

    def valid(self, valid_l_loader):
        valid_epoch = len(valid_l_loader)
        with torch.no_grad():
            self.ema.model.eval()
            valid_dataset = iter(valid_l_loader)
            valid_conf_mat = ConfMatrix(self.data_loader.num_segments)
            for i in range(valid_epoch):
                valid_data, valid_label = valid_dataset.next()
                valid_data, valid_label = valid_data.to(self.device), valid_label.to(self.device)

                pred, _ = self.ema.model(valid_data)
                pred_u_large_raw = F.interpolate(pred, size=valid_label.shape[1:], mode='bilinear', align_corners=True)
                valid_conf_mat.update(pred_u_large_raw.argmax(1).flatten(), valid_label.flatten())
        self.mIoU, _ = valid_conf_mat.get_metrics()
        self.score_dict['mIoU'] = self.mIoU

    def inference(self, test_loader, save_path, sample_submission):
        # batch size of the test loader should be 1
        class_map = {0:'ship', 1:'container_truck', 2:'forklift', 3:'reach_stacker'}

        test_epoch = len(test_loader)
        file_names = []
        classes = []
        predictions = []
        with torch.no_grad():
            self.ema.model.eval()
            test_dataset = iter(test_loader)
            for i in tqdm(range(test_epoch)):
                test_data, img_size, filename = test_dataset.next()
                test_data = test_data.to(self.device)
                pred, _ = self.ema.model(test_data)
                pred_u_large_raw = F.interpolate(pred, size=img_size[0].tolist(), mode='bilinear', align_corners=True)
                class_num = pred_u_large_raw[0].sum(dim=(1,2))[1:].argmax().item()
                class_of_image = class_map[class_num]
                class_mask = (pred_u_large_raw[0][class_num + 1] -  pred_u_large_raw[0][0] > 0).int().cpu().numpy()
                coverted_coordinate = mask_to_coordinates(class_mask)
                file_names.append(filename[0])
                classes.append(class_of_image)
                predictions.append(coverted_coordinate)

        submission_df = pd.DataFrame({'file_name':file_names, 'class':classes, 'prediction':predictions})
        submission_df = pd.merge(sample_submission['file_name'], submission_df, left_on='file_name', right_on='file_name', how='left')
        submission_df.to_csv(save_path, index=False, encoding='utf-8')

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.mIoU = 0
        self.score_dict = dict()
        self.elapsed_time = 0