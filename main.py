import os
import sys

import ray

from data import load_dataset
from trainer import get_trainer
from utils.utils import print_eval_acc, print_train_acc, load_with_default_yaml, save_dict_as_one_line_csv


def main(working_dir, seed, train_epochs, eval_every, use_ray, ray_params, data_params, trainer_params):
    if use_ray:
        ray.init(**ray_params)

    (train_iterator, test_iterator), metadata = load_dataset(**data_params)
    trainer = get_trainer(seed=seed, train_iterator=train_iterator, test_iterator=test_iterator, metadata=metadata,
                          **trainer_params)

    eval_metrics = trainer.evaluate()
    print_eval_acc(eval_metrics)

    for i in range(train_epochs):
        train_metrics = trainer.train_epoch()
        print_train_acc(train_metrics, epoch=i)
        if eval_every is not None and (i + 1) % eval_every == 0:
            eval_metrics = trainer.evaluate()
            print_eval_acc(eval_metrics)

    eval_metrics = trainer.evaluate()
    print_eval_acc(eval_metrics)

    if use_ray:
        ray.shutdown()
    metrics = dict(**train_metrics, **eval_metrics)
    save_dict_as_one_line_csv(metrics, filename=os.path.join(working_dir, "metrics.csv"))
    return metrics


if __name__ == "__main__":
    param_path = sys.argv[1]
    param_dict = load_with_default_yaml(path=param_path)
    main(**param_dict)
