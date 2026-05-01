"""Tests for the standalone SLURM submission helper."""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from fdq_submit import (
    build_parameter_study_runs,
    create_submit_file,
    find_parameter_ranges,
    load_conf_file,
    print_submission_summary,
)

try:
    import yaml  # noqa: F401

    HAS_YAML = True
except ModuleNotFoundError:
    HAS_YAML = False


class TestFdqSubmit(unittest.TestCase):
    """Unit tests for fdq_submit.py helpers."""

    def test_create_submit_file_preserves_package_index_urls(self):
        """The generated submit script must keep https:// URLs intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            submit_path = os.path.join(temp_dir, "job.submit")
            job_config = {
                "user": "tester",
                "job_time": 1,
                "ntasks": 1,
                "cpus_per_task": 1,
                "cpus_per_task_test": 1,
                "nodes": 1,
                "nodelist": "None",
                "gres": "gpu:1",
                "gres_test": "gpu:1",
                "mem": "1G",
                "mem_test": "1G",
                "partition": "gpu",
                "account": "account",
                "run_train": True,
                "run_test": False,
                "is_test": False,
                "job_tag": "_train",
                "auto_resubmit": False,
                "resume_chpt_path": "None",
                "log_path": temp_dir,
                "stop_grace_time": 5,
                "python_env_module": "python/3.12",
                "uv_env_module": "uv/0.6",
                "cuda_env_module": "None",
                "fdq_version": "0.0.77",
                "fdq_test_repo": True,
                "config_path": temp_dir,
                "config_name": "experiment",
                "scratch_results_path": "/scratch/fdq_results/",
                "scratch_data_path": "/scratch/fdq_data/",
                "results_path": temp_dir,
                "submit_file_path": submit_path,
            }

            create_submit_file(job_config, {"additional_pip_packages": None}, submit_path)

            with open(submit_path, encoding="utf8") as submit_file:
                content = submit_file.read()

            self.assertIn("https://test.pypi.org/simple/", content)
            self.assertIn("https://pypi.org/simple", content)
            self.assertIn('PARAMETER_OVERRIDES=""', content)
            self.assertIn('export FDQ_PARAMETER_RUN_TAG=""', content)
            self.assertIn('export FDQ_PARAMETER_STUDY_PATHS=""', content)
            self.assertIn('export FDQ_TEST_RESULTS_DIR=""', content)
            self.assertNotIn("#parameter_overrides#", content)
            self.assertNotIn("#parameter_run_tag#", content)
            self.assertNotIn("#parameter_study_paths#", content)
            self.assertNotIn("#test_results_dir#", content)

    def test_create_submit_file_includes_parameter_overrides(self):
        """Generated submit scripts pass concrete parameter-study overrides to fdq."""
        with tempfile.TemporaryDirectory() as temp_dir:
            submit_path = os.path.join(temp_dir, "job.submit")
            job_config = {
                "user": "tester",
                "job_time": 1,
                "ntasks": 1,
                "cpus_per_task": 1,
                "cpus_per_task_test": 1,
                "nodes": 1,
                "nodelist": "None",
                "gres": "gpu:1",
                "gres_test": "gpu:1",
                "mem": "1G",
                "mem_test": "1G",
                "partition": "gpu",
                "account": "account",
                "run_train": True,
                "run_test": False,
                "is_test": False,
                "job_tag": "_train_p001",
                "auto_resubmit": False,
                "resume_chpt_path": "None",
                "log_path": temp_dir,
                "stop_grace_time": 5,
                "python_env_module": "python/3.12",
                "uv_env_module": "uv/0.6",
                "cuda_env_module": "None",
                "fdq_version": "0.0.77",
                "fdq_test_repo": True,
                "config_path": temp_dir,
                "config_name": "experiment",
                "scratch_results_path": "/scratch/fdq_results/",
                "scratch_data_path": "/scratch/fdq_data/",
                "results_path": temp_dir,
                "submit_file_path": submit_path,
                "parameter_overrides": "models.simpleNet.optimizer.args.lr=0.001",
                "parameter_run_tag": "_p001",
                "parameter_study_paths": "models.simpleNet.optimizer.args.lr",
            }

            create_submit_file(job_config, {"additional_pip_packages": None}, submit_path)

            with open(submit_path, encoding="utf8") as submit_file:
                content = submit_file.read()

            self.assertIn('PARAMETER_OVERRIDES="models.simpleNet.optimizer.args.lr=0.001"', content)
            self.assertIn('export FDQ_PARAMETER_RUN_TAG="_p001"', content)
            self.assertIn('export FDQ_PARAMETER_STUDY_PATHS="models.simpleNet.optimizer.args.lr"', content)
            self.assertIn('export FDQ_TEST_RESULTS_DIR=""', content)
            self.assertIn("$PARAMETER_OVERRIDES mode.run_test_auto=false", content)
            self.assertIn(
                'TRAINED_RESULTS_DIR=$(find "$SCRATCH_RESULTS_PATH" -type d -name "*__${SLURM_JOB_ID}${FDQ_PARAMETER_RUN_TAG}"',
                content,
            )
            self.assertIn('export FDQ_TEST_RESULTS_DIR="$TRAINED_RESULTS_DIR"', content)
            self.assertIn(
                r"s|^\(#SBATCH --output=.*\)_train\([^/[:space:]]*\.out\)|\1_test\2|g",
                content,
            )
            self.assertIn(
                r"s|^\(#SBATCH --error=.*\)_train\([^/[:space:]]*\.err\)|\1_test\2|g",
                content,
            )

    def test_parameter_study_expands_range_product(self):
        """Range markers expand to every parameter combination."""
        cfg = {
            "models": {
                "simpleNet": {
                    "optimizer": {
                        "args": {
                            "lr": ["0.001:0.005:5"],
                        }
                    }
                }
            },
            "train": {"args": {"epochs": ["1:3:3"]}},
        }

        ranges = find_parameter_ranges(cfg)
        runs = build_parameter_study_runs(cfg, parameter_study_enabled=True)

        self.assertEqual(
            ranges,
            [
                ("models.simpleNet.optimizer.args.lr", ["0.001", "0.002", "0.003", "0.004", "0.005"]),
                ("train.args.epochs", ["1", "2", "3"]),
            ],
        )
        self.assertEqual(len(runs), 15)
        self.assertEqual(
            runs[0][1],
            "models.simpleNet.optimizer.args.lr=0.001 train.args.epochs=1",
        )
        self.assertEqual(
            runs[-1][1],
            "models.simpleNet.optimizer.args.lr=0.005 train.args.epochs=3",
        )

    def test_parameter_study_disabled_uses_first_values(self):
        """Disabled parameter studies keep one job with the first value from each range."""
        cfg = {
            "models": {"simpleNet": {"optimizer": {"args": {"lr": ["0.001:0.005:5"]}}}},
            "train": {"args": {"epochs": ["1:3:3"]}},
        }

        runs = build_parameter_study_runs(cfg, parameter_study_enabled=False)

        self.assertEqual(len(runs), 1)
        self.assertEqual(
            runs[0][1],
            "models.simpleNet.optimizer.args.lr=0.001 train.args.epochs=1",
        )
        self.assertEqual(runs[0][0]["models"]["simpleNet"]["optimizer"]["args"]["lr"], "0.001")
        self.assertEqual(runs[0][0]["train"]["args"]["epochs"], "1")

    def test_parameter_study_expands_categorical_values(self):
        """Categorical markers expand to every listed non-numeric value."""
        cfg = {
            "data": {"OXPET": {"args": {"shuffle_train": ["true:false"]}}},
            "models": {"simpleNet": {"optimizer": {"class_name": [{"torch.optim.Adam": "torch.optim.SGD"}]}}},
        }

        ranges = find_parameter_ranges(cfg)
        runs = build_parameter_study_runs(cfg, parameter_study_enabled=True)

        self.assertEqual(
            ranges,
            [
                ("data.OXPET.args.shuffle_train", ["true", "false"]),
                ("models.simpleNet.optimizer.class_name", ["torch.optim.Adam", "torch.optim.SGD"]),
            ],
        )
        self.assertEqual(len(runs), 4)
        self.assertEqual(
            runs[0][1],
            "data.OXPET.args.shuffle_train=true models.simpleNet.optimizer.class_name=torch.optim.Adam",
        )
        self.assertEqual(
            runs[-1][1],
            "data.OXPET.args.shuffle_train=false models.simpleNet.optimizer.class_name=torch.optim.SGD",
        )

    def test_submission_summary_clearly_marks_parameter_study(self):
        """The success block clearly states when a parameter study was submitted."""
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            print_submission_summary(
                submitted_jobs=[
                    ("123", "/tmp/run_001.submit", "models.simpleNet.optimizer.args.lr=0.001"),
                    ("124", "/tmp/run_002.submit", "models.simpleNet.optimizer.args.lr=0.002"),
                ],
                config_name="experiment",
                config_path="/tmp",
                last_job_config={"results_path": "/tmp/results", "log_path": "/tmp/logs"},
                parameter_ranges=[
                    ("models.simpleNet.optimizer.args.lr", ["0.001", "0.002"]),
                ],
                parameter_study=True,
            )

        summary = stdout.getvalue()

        self.assertIn("Parameter Study: enabled", summary)
        self.assertIn("Parameter Runs:  2", summary)
        self.assertIn("Sweep Params:    models.simpleNet.optimizer.args.lr", summary)
        self.assertIn("Submitted Jobs:  2", summary)

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_parameter_study_ranges_are_found_in_parent_defaults(self):
        """Range markers inherited from parent configs still produce scalar overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = os.path.join(temp_dir, "parent.yaml")
            child_path = os.path.join(temp_dir, "child.yaml")

            with open(parent_path, "w", encoding="utf8") as parent_file:
                parent_file.write(
                    "mode:\n"
                    "  parameter_study: false\n"
                    "models:\n"
                    "  simpleNet:\n"
                    "    optimizer:\n"
                    "      args:\n"
                    "        lr: [0.001:0.005:5]\n"
                )

            with open(child_path, "w", encoding="utf8") as child_file:
                child_file.write("defaults:\n  - parent\n  - _self_\n")

            cfg = load_conf_file(child_path)
            runs = build_parameter_study_runs(cfg, parameter_study_enabled=cfg["mode"]["parameter_study"])

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0][1], "models.simpleNet.optimizer.args.lr=0.001")

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_integer_parameter_ranges_are_loaded_as_ranges(self):
        """Integer ranges like [4:6:2] are not parsed as YAML sexagesimal ints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "experiment.yaml")

            with open(config_path, "w", encoding="utf8") as config_file:
                config_file.write(
                    "mode:\n"
                    "  parameter_study: true\n"
                    "train:\n"
                    "  args:\n"
                    "    epochs: [4:6:2]\n"
                    "models:\n"
                    "  simpleNet:\n"
                    "    optimizer:\n"
                    "      args:\n"
                    "        lr: [0.001:0.002:2]\n"
                )

            cfg = load_conf_file(config_path)
            ranges = find_parameter_ranges(cfg)
            runs = build_parameter_study_runs(cfg, parameter_study_enabled=cfg["mode"]["parameter_study"])

            self.assertIn(("train.args.epochs", ["4", "6"]), ranges)
            self.assertEqual(len(runs), 4)
            self.assertIn("train.args.epochs=4", runs[0][1])
            self.assertIn("train.args.epochs=6", runs[-1][1])

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_categorical_parameter_studies_are_loaded_from_yaml(self):
        """Categorical study markers survive YAML loading in both supported forms."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "experiment.yaml")

            with open(config_path, "w", encoding="utf8") as config_file:
                config_file.write(
                    "mode:\n"
                    "  parameter_study: true\n"
                    "data:\n"
                    "  OXPET:\n"
                    "    args:\n"
                    "      shuffle_train: [true:false]\n"
                    "models:\n"
                    "  simpleNet:\n"
                    "    optimizer:\n"
                    '      class_name: ["torch.optim.Adam":"torch.optim.SGD"]\n'
                )

            cfg = load_conf_file(config_path)
            ranges = find_parameter_ranges(cfg)
            runs = build_parameter_study_runs(cfg, parameter_study_enabled=cfg["mode"]["parameter_study"])

            self.assertIn(("data.OXPET.args.shuffle_train", ["true", "false"]), ranges)
            self.assertIn(("models.simpleNet.optimizer.class_name", ["torch.optim.Adam", "torch.optim.SGD"]), ranges)
            self.assertEqual(len(runs), 4)

    @unittest.skipUnless(HAS_YAML, "PyYAML is not installed")
    def test_load_conf_file_merges_hydra_defaults(self):
        """fdq_submit uses inherited config values the same way Hydra configs do."""
        with tempfile.TemporaryDirectory() as temp_dir:
            parent_path = os.path.join(temp_dir, "parent.yaml")
            child_path = os.path.join(temp_dir, "child.yaml")

            with open(parent_path, "w", encoding="utf8") as parent_file:
                parent_file.write(
                    "store:\n"
                    "  results_path: /tmp/fdq_results\n"
                    "slurm_cluster:\n"
                    "  partition: gpu\n"
                    "  account: base\n"
                    "mode:\n"
                    "  run_train: true\n"
                )

            with open(child_path, "w", encoding="utf8") as child_file:
                child_file.write(
                    "defaults:\n"
                    "  - parent\n"
                    "  - _self_\n"
                    "slurm_cluster:\n"
                    "  account: child\n"
                    "mode:\n"
                    "  run_test_auto: true\n"
                )

            cfg = load_conf_file(child_path)

            self.assertEqual(cfg["store"]["results_path"], "/tmp/fdq_results")
            self.assertEqual(cfg["slurm_cluster"]["partition"], "gpu")
            self.assertEqual(cfg["slurm_cluster"]["account"], "child")
            self.assertTrue(cfg["mode"]["run_train"])
            self.assertTrue(cfg["mode"]["run_test_auto"])


if __name__ == "__main__":
    unittest.main()
