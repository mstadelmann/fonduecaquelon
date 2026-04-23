"""Tests for the standalone SLURM submission helper."""

import os
import tempfile
import unittest

from fdq_submit import create_submit_file, load_conf_file

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
                "fdq_version": "0.0.76",
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
