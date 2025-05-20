import unittest
import yaml
from pathlib import Path
import tempfile
import os

# Adjust the import path based on your project structure.
# If tests/ is a top-level directory and fuchsia/ is also top-level,
# and assuming your PYTHONPATH is set up correctly or you're running tests
# from the project root, this might work.
# You might need to add fuchsia to sys.path or use relative imports if running as a module.
from fuchsia.cli import create_default_config, get_default_config_dict

class TestCli(unittest.TestCase):

    def test_create_default_config_output(self):
        """
        Tests that create_default_config generates a YAML file with the expected content.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output_path = Path(tmpdir) / "test_config.yaml"
            
            # Call the function to create the config file
            create_default_config(output_path=str(tmp_output_path))
            
            # 1. Check if the file was created
            self.assertTrue(os.path.exists(tmp_output_path), "Config file was not created.")
            
            # 2. Read the generated file and load its YAML content
            with open(tmp_output_path, 'r') as f:
                generated_config = yaml.safe_load(f)
            
            # 3. Get the expected config dictionary
            expected_config = get_default_config_dict()
            
            # 4. Compare the loaded config with the expected config
            # This will compare structure and values.
            # PyYAML will create new Python objects when loading, so direct object
            # comparison for aliased values isn't the goal here, but rather
            # that the values themselves are correct and present where expected.
            self.assertEqual(generated_config, expected_config, 
                             "Generated config content does not match expected content.")

    def test_yaml_anchors_and_aliases_manually(self):
        """
        Tests if specific values in the generated YAML are aliases of others,
        simulating the effect of anchors. This requires a deeper inspection
        of the generated YAML content if we can't rely on PyYAML's loader
        preserving anchor information directly in the loaded dict for easy check.

        However, PyYAML's standard `safe_load` resolves aliases to their referenced
        values. So, `generated_config` will have the actual values, not alias objects.
        The check in `test_create_default_config_output` already ensures these resolved
        values match the `expected_config`.

        If the requirement were to check that the *YAML text itself* contains `*` and `&`
        syntax, we would need to parse the raw YAML string. But comparing the loaded
        Python dicts (as done above) effectively checks that the aliasing worked
        in terms of the final values.

        This test will verify that by checking specific known aliased paths point to
        the same values as their anchors.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_output_path = Path(tmpdir) / "test_config_anchor.yaml"
            create_default_config(output_path=str(tmp_output_path))
            
            with open(tmp_output_path, 'r') as f:
                generated_config = yaml.safe_load(f)

            # Check that values that should be aliased are indeed equal after loading
            self.assertEqual(generated_config["model"]["max_model_len"], generated_config["generation"]["max_len"])
            self.assertEqual(generated_config["grpo"]["group_size"], generated_config["generation"]["group_size"])
            self.assertEqual(generated_config["server"]["generation_batch_size"], generated_config["generation"]["batch_size"])
            
            self.assertEqual(generated_config["server"]["vllm"]["max_tokens"], generated_config["generation"]["max_len"])
            self.assertEqual(generated_config["server"]["vllm"]["n"], generated_config["generation"]["group_size"])
            self.assertEqual(generated_config["server"]["vllm"]["temperature"], generated_config["generation"]["temperature"])
            self.assertEqual(generated_config["server"]["vllm"]["top_p"], generated_config["generation"]["top_p"])
            self.assertEqual(generated_config["server"]["vllm"]["top_k"], generated_config["generation"]["top_k"])
            self.assertEqual(generated_config["server"]["vllm"]["min_p"], generated_config["generation"]["min_p"])


if __name__ == '__main__':
    unittest.main()
