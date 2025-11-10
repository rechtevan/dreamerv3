"""
Tests for dreamerv3.main - Entry point and configuration

Coverage goal: 90% (starting with ~20-30% foundational tests)
"""

import pathlib

import elements
import pytest
import ruamel.yaml as yaml


class TestConfiguration:
    """Tests for configuration loading and parsing"""

    def test_configs_file_exists(self):
        """Test that configs.yaml file exists"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        assert config_path.exists(), "configs.yaml should exist"

    def test_configs_yaml_valid(self):
        """Test that configs.yaml is valid YAML"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        assert isinstance(parsed, dict), "configs.yaml should parse to dict"
        assert "defaults" in parsed, "configs.yaml should have 'defaults' section"

    def test_default_config_structure(self):
        """Test default configuration has required keys"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        defaults = parsed["defaults"]

        # Check essential configuration keys exist
        required_keys = {
            "logdir",
            "script",
            "batch_size",
            "batch_length",
            "run",
            "jax",
            "agent",
            "replay",
            "env",
        }
        assert required_keys.issubset(set(defaults.keys())), (
            f"Missing required keys in defaults: {required_keys - set(defaults.keys())}"
        )

    def test_agent_config_structure(self):
        """Test agent configuration has required components"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        agent_config = parsed["defaults"]["agent"]

        # Check essential agent configuration
        required_agent_keys = {"dyn", "enc", "dec", "loss_scales", "opt"}
        assert required_agent_keys.issubset(set(agent_config.keys())), (
            f"Missing required agent keys: {required_agent_keys - set(agent_config.keys())}"
        )

        # Check dyn (dynamics/RSSM) configuration
        assert "typ" in agent_config["dyn"]
        assert agent_config["dyn"]["typ"] in ["rssm"]

        # Check enc (encoder) configuration
        assert "typ" in agent_config["enc"]
        assert agent_config["enc"]["typ"] in ["simple"]

        # Check dec (decoder) configuration
        assert "typ" in agent_config["dec"]
        assert agent_config["dec"]["typ"] in ["simple"]

    def test_loss_scales_configuration(self):
        """Test loss scales are properly configured"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        loss_scales = parsed["defaults"]["agent"]["loss_scales"]

        expected_scales = {
            "rec",
            "rew",
            "con",
            "dyn",
            "rep",
            "policy",
            "value",
            "repval",
        }
        assert set(loss_scales.keys()) == expected_scales, (
            f"Loss scales mismatch. Expected: {expected_scales}, Got: {set(loss_scales.keys())}"
        )

        # All loss scales should be numeric
        assert all(isinstance(v, (int, float)) for v in loss_scales.values())

    def test_batch_configuration(self):
        """Test batch size and length are valid"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        defaults = parsed["defaults"]

        assert isinstance(defaults["batch_size"], int)
        assert defaults["batch_size"] > 0
        assert isinstance(defaults["batch_length"], int)
        assert defaults["batch_length"] > 0

    def test_jax_configuration(self):
        """Test JAX configuration is valid"""
        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        jax_config = parsed["defaults"]["jax"]

        assert "platform" in jax_config
        assert jax_config["platform"] in ["cpu", "cuda", "tpu"]
        assert "compute_dtype" in jax_config
        assert "jit" in jax_config
        assert isinstance(jax_config["jit"], bool)


class TestScriptSelection:
    """Tests for script selection logic"""

    def test_valid_scripts(self):
        """Test that valid script modes are recognized"""
        valid_scripts = ["train", "train_eval", "eval_only"]

        config_path = pathlib.Path(__file__).parent.parent / "configs.yaml"
        configs = config_path.read_text()
        parsed = yaml.YAML(typ="safe").load(configs)
        default_script = parsed["defaults"]["script"]

        assert default_script in valid_scripts, (
            f"Default script '{default_script}' should be one of {valid_scripts}"
        )
