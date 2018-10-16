import json
import numpy as np
import os
import shutil
import warnings

import keras.backend as kb
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

class MultiClassAUROC(Callback):
    """Class to monitor the Area Under Receiver Operator Curve (AUROC) and
    update the model.
    
    Arguments:
        Callback {Callback} -- Keras built-in callback
    """
    def __init__(self, sequence, class_names, weights_path, 
                stats=None, workers=1):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            f"best_{os.path.split(weights_path)[1]}",
        )
        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log"
        )
        self.stats_output_path = os.path.join(
            os.path.splot(weihts_path)[0],
            ".training_stats.json"
        )

        # To resume training a model
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}
        
        # area under receiver operator characteristic
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []
                
    def on_end_epoch(self, epoch: int, logs={}):
        """
        Calculate the mean AUROC and save the best parameters accordingly
        
        Arguments:
            epoch {int} -- Current epoch number
        
        Keyword Arguments:
            logs {[type]} -- [Dictionary of logs] (default: {})
        """

        y_hat = self.model.predict_generator(self.sequence, workers=self.workers)
        y = self.sequence.get_y_true()

        print(f"Epoch #{epoch + 1} Area Under ROC")
        current_auroc = []
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except:
                score = 0
                current_auroc.append(score)
                print(f"{i + 1}. {self.class_names[i]}: {score}")
            print("***********************************")

            mean_auroc = np.mean(current_auroc)
            print(f"Mean AUROC : {mean_auroc}")
            if mean_auroc > self.stats["best_mean_auroc"]:
                print(f"Mean AUROC Increased! Updating best AUROC from {self.stats['best_mean_auroc']} to {mean_auroc}")

                # copy the best model
                shutil.copy(self.weights_path, self.best_weights_path)

                # update log files accordingly
                print(f"Updating the log file! : {self.best_auroc_log_path}")
                with open(self.best_auroc_log_path, "a") as f:
                    f.write(f"(Epoch # {epoch + 1}) AUROC : {mean_auroc}, Learning Rate : {self.stats['lr']}\n")
                
                # write stats output
                # used for resuming model training
                with open(self.stats_output_path, 'W') as f:
                    json.dump(self.stats, f)

                print(f"Update model file : {self.weights_path} --> {self.best_weights_path}")
                self.stats["best_mean_auroc"] = mean_auroc
                print("**********************************")

            return
    
class MultiGPUModelCheckpoint(Callback):
    """
    Checkpointing for multi_gpu_model training

    reference : https://github.com/keras-team/keras/issues/8463

    Arguments:
        Callback {[type]} -- [description]
    """

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False, mode='auto',
                 period=1):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.base_model = base_model
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.save_weights_only = save_weights_only
        self.mode = mode

        # track last saving of parameters over epoch count
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                        'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf
    
    def on_end_epoch(self, epoch, logs=None):
        """
        Method containing callbacks for saving model parameters and other functionality
        that one would like to perform at the end of a given training epoch.
        
        Arguments:
            epoch {[type]} -- [description]
        
        Keyword Arguments:
            logs {[type]} -- [description] (default: {None})
        """

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.05f to %0.05f,'
                            ' saving model to %s'
                            % (epoch + 1, self.monitor, self.best, 
                               current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: Saving Model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)