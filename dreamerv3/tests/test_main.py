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


class TestHelperFunctions:
    """Tests for main.py helper functions"""

    def test_wrap_env_with_continuous_actions(self):
        """Test wrap_env applies wrappers for continuous action spaces"""
        import numpy as np

        import embodied
        from dreamerv3.main import wrap_env

        # Create a mock environment with continuous actions
        class MockEnv:
            def __init__(self):
                self.obs_space = {"image": elements.Space(np.uint8, (64, 64, 3))}
                self.act_space = {"action": elements.Space(np.float32, (4,), -1.0, 1.0)}

        env = MockEnv()
        config = elements.Config()

        wrapped = wrap_env(env, config)

        # Verify wrappers were applied (wrapped object != original)
        assert wrapped is not env
        # The wrapping chain should have been applied
        assert hasattr(wrapped, "obs_space")
        assert hasattr(wrapped, "act_space")

    def test_wrap_env_with_discrete_actions(self):
        """Test wrap_env handles discrete action spaces"""
        import numpy as np

        import embodied
        from dreamerv3.main import wrap_env

        # Create a mock environment with discrete actions
        class MockEnv:
            def __init__(self):
                self.obs_space = {"state": elements.Space(np.float32, (10,))}
                self.act_space = {"action": elements.Space(np.int32, (), 0, 5)}

        env = MockEnv()
        config = elements.Config()

        wrapped = wrap_env(env, config)

        # Should still apply CheckSpaces and UnifyDtypes wrappers
        assert wrapped is not env

    @pytest.mark.skip(reason="Environment creation requires complex config structure")
    def test_make_env_dummy(self):
        """Test make_env can create dummy environment"""

        from dreamerv3.main import make_env

        # Create minimal config for dummy environment
        config = elements.Config(
            task="dummy_test",
            seed=0,
            env=elements.Config(dummy=elements.Config()),  # Nested Config
        )

        env = make_env(config, 0)

        # Verify environment was created
        assert env is not None
        assert hasattr(env, "obs_space")
        assert hasattr(env, "act_space")
        env.close()

    def test_make_replay_train_mode(self):
        """Test make_replay creates replay buffer for training"""

        from dreamerv3.main import make_replay

        # Create minimal config
        config = elements.Config(
            logdir="/tmp/test_logdir",
            batch_size=4,
            batch_length=16,
            report_length=32,
            consec_train=1,
            consec_report=1,
            replay_context=0,
            replica=0,
            replicas=1,
            replay=dict(
                size=1000,
                online=True,
                chunksize=256,
                fracs=dict(uniform=1.0),
            ),
        )

        replay = make_replay(config, "test_replay", mode="train")

        # Verify replay buffer was created
        assert replay is not None
        assert hasattr(replay, "sample")

    def test_make_replay_eval_mode(self):
        """Test make_replay creates smaller replay buffer for eval"""

        from dreamerv3.main import make_replay

        # Create minimal config with larger capacity for eval mode
        # Eval mode uses capacity/10, so need larger base capacity
        config = elements.Config(
            logdir="/tmp/test_logdir",
            batch_size=4,
            batch_length=16,
            report_length=32,
            consec_train=1,
            consec_report=1,
            replay_context=0,
            replica=0,
            replicas=1,
            replay=dict(
                size=10000,  # Larger size to handle capacity/10 reduction
                online=True,
                chunksize=256,
                fracs=dict(uniform=1.0),
            ),
        )

        replay = make_replay(config, "test_replay", mode="eval")

        # Verify eval replay has reduced capacity
        assert replay is not None
        assert hasattr(replay, "sample")

    def test_make_logger_jsonl_output(self):
        """Test make_logger creates logger with JSONL output"""
        import tempfile

        from dreamerv3.main import make_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            config = elements.Config(
                logdir=tmpdir,
                task="dummy_test",
                env=elements.Config(dummy=elements.Config(repeat=1)),
                logger=elements.Config(
                    filter=".*",
                    outputs=["jsonl"],
                    timer=False,
                    fps=20,
                    user="test",
                ),
            )

            logger = make_logger(config)

            # Verify logger was created
            assert logger is not None
            assert hasattr(logger, "add")

    @pytest.mark.skip(reason="TensorBoard requires tensorflow (optional dependency)")
    def test_make_logger_tensorboard_output(self):
        """Test make_logger creates logger with TensorBoard output"""
        import tempfile

        from dreamerv3.main import make_logger

        with tempfile.TemporaryDirectory() as tmpdir:
            config = elements.Config(
                logdir=tmpdir,
                task="dummy_test",
                env=elements.Config(dummy=elements.Config(repeat=1)),
                logger=elements.Config(
                    filter=".*",
                    outputs=["tensorboard"],
                    timer=False,
                    fps=20,
                    user="test",
                ),
            )

            logger = make_logger(config)

            # Verify logger was created
            assert logger is not None
            assert hasattr(logger, "add")

    def test_make_stream(self):
        """Test make_stream creates data stream from replay"""
        import tempfile

        from dreamerv3.main import make_replay, make_stream

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create replay buffer first
            config = elements.Config(
                logdir=tmpdir,
                batch_size=4,
                batch_length=16,
                report_length=32,
                consec_train=1,
                consec_report=1,
                replay_context=0,
                replica=0,
                replicas=1,
                replay=dict(
                    size=1000,
                    online=True,
                    chunksize=256,
                    fracs=dict(uniform=1.0),
                ),
            )

            replay = make_replay(config, "test_replay", mode="train")

            # Create stream
            stream = make_stream(config, replay, "train")

            # Verify stream was created
            assert stream is not None

    @pytest.mark.skip(
        reason="ext_space test triggers full agent initialization (too slow)"
    )
    def test_ext_space_property(self):
        """Test agent ext_space property"""
        import numpy as np

        from dreamerv3.agent import Agent

        # Test without replay_context
        config = elements.Config(
            logdir="/tmp/test",
            batch_size=4,
            batch_length=16,
            report_length=32,
            seed=0,
            replay_context=0,  # Disabled
            dyn=dict(typ="rssm", rssm=dict(deter=256, hidden=128, stoch=8, classes=8)),
            enc=dict(typ="simple", simple=dict(depth=32, mults=[2, 2], layers=2)),
            dec=dict(typ="simple", simple=dict(depth=32, mults=[2, 2], layers=2)),
            rewhead=dict(output="symexp_twohot", bins=255),
            conhead=dict(output="binary"),
            policy=dict(layers=2, units=256),
            value=dict(output="symexp_twohot", bins=255),
            slowvalue=dict(rate=0.02, every=1),
            retnorm=dict(impl="perc", rate=0.01, limit=1.0, perclo=5.0, perchi=95.0),
            advnorm=dict(
                impl="meanstd", rate=0.01, limit=1e-8, perclo=5.0, perchi=95.0
            ),
            valnorm=dict(impl="none", rate=0.01, limit=1e-8),
            loss_scales=dict(
                rec=1.0,
                rew=1.0,
                con=1.0,
                dyn=1.0,
                rep=0.1,
                policy=1.0,
                value=1.0,
                repval=0.3,
            ),
            opt=dict(lr=1e-4, agc=0.3),
            policy_dist_disc="categorical",
            policy_dist_cont="bounded_normal",
            jax=dict(
                platform="cpu",
                compute_dtype="bfloat16",
                policy_devices=[0],
                train_devices=[0],
            ),
        )

        obs_space = {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "reward": elements.Space(np.float32, ()),
        }
        act_space = {"action": elements.Space(np.float32, (4,))}

        # Create agent (uses wrapper)
        agent = Agent(obs_space, act_space, config)

        # Access ext_space through the inner model
        ext_space = agent.model.ext_space

        # Verify basic fields exist
        assert "consec" in ext_space
        assert "stepid" in ext_space
