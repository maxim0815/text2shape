from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class Evaluation:

    def __init__(self, store_dir, name, stats, hyper_parameters = 0):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        self.tf_writer = SummaryWriter(os.path.join(
            store_dir, "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S"))))

        self.stats = stats

        if hyper_parameters != 0:
            self.tf_writer.add_hparams(hyper_parameters, {})


    def write_episode_data(self, episode, eval_dict):
        """
         Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
         e.g. eval_dict = {"loss" : 1e-4}
        """
        for key in eval_dict:
            assert(key in self.stats)
            self.tf_writer.add_scalar(key, eval_dict[key], episode)


    def close_session(self):
        self.tf_writer.close()
