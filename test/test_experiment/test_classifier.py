import argparse
import json
import os
import unittest
import glob

from fdq.experiment import fdqExperiment
from fdq.testing import run_test, find_model_path, find_experiment_result_dirs


class TestMNISTClassifier(unittest.TestCase):
    def test_run_train(self):
        exp_path = os.path.join(
            os.path.split(os.path.abspath(__file__))[0],
            "mnist_testexp_dense.json",
        )

        args = argparse.Namespace(
            experimentfile=exp_path,
            train_model=True,
            test_model_ia=False,
            test_model_auto=False,
            dump_model=False,
            print_model=False,
            resume_path=None,
        )

        experiment = fdqExperiment(args)
        experiment.mode.unittest()
        experiment.prepareTraining()
        experiment.trainer.train(experiment)

        res_dir = experiment.results_dir

        # check if infofile exists (did experiment start at all?)
        info_file = os.path.join(res_dir, "info.json")
        self.assertTrue(os.path.exists(info_file))

        # check that loss is going down
        history_file = os.path.join(res_dir, "history.json")
        with open(history_file, encoding="utf8") as json_file:
            history = json.load(json_file)
            self.assertTrue(history["train"][0] > history["train"][-1])

        run_test(experiment)

        res_d, _ = find_model_path(experiment)

        # check if test results file exists
        res_paths = glob.glob(res_d + "/test/*/00_test_results_*")
        self.assertTrue(len(res_paths) > 0)

        with open(res_paths[0], encoding="utf8") as json_file:
            testres = json.load(json_file)
            self.assertTrue(testres["test results"] > 0.2)
