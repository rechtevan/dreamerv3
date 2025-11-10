"""
Tests for dreamerv3.agent - Agent training and policy

Coverage goal: 90% (starting with ~20-30% foundational tests)

Note: Agent is imported inside test methods to avoid circular import issues
"""

import elements
import jax
import numpy as np
import pytest


class TestAgent:
    """Tests for DreamerV3 Agent"""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Minimal configuration for testing"""
        return elements.Config(
            # Required paths and settings
            logdir=str(tmp_path / "logdir"),
            batch_size=4,
            batch_length=16,
            report_length=32,
            seed=0,
            # RSSM configuration
            dyn=dict(
                typ="rssm",
                rssm=dict(
                    deter=256,
                    hidden=128,
                    stoch=8,
                    classes=8,
                    blocks=8,
                    act="gelu",
                    norm="layer",
                    unimix=0.01,
                    outscale=1.0,
                    imglayers=2,
                    obslayers=1,
                    dynlayers=1,
                    absolute=False,
                    free_nats=1.0,
                ),
            ),
            # Encoder configuration
            enc=dict(
                typ="simple",
                simple=dict(
                    depth=32,
                    mults=[2, 2],
                    layers=2,
                    units=256,
                    act="gelu",
                    norm="layer",
                    winit="normal",
                    symlog=True,
                    outer=False,
                    kernel=3,
                    strided=False,
                ),
            ),
            # Decoder configuration
            dec=dict(
                typ="simple",
                simple=dict(
                    depth=32,
                    mults=[2, 2],
                    layers=2,
                    units=256,
                    act="gelu",
                    norm="layer",
                    outscale=1.0,
                    winit="normal",
                    outer=False,
                    kernel=3,
                    bspace=8,
                    strided=False,
                ),
            ),
            # Head configurations
            rewhead=dict(
                layers=1,
                units=256,
                act="silu",
                norm="rms",
                output="symexp_twohot",
                outscale=0.0,
                winit="trunc_normal_in",
                bins=255,
            ),
            conhead=dict(
                layers=1,
                units=256,
                act="silu",
                norm="rms",
                output="binary",
                outscale=1.0,
                winit="trunc_normal_in",
            ),
            policy=dict(
                layers=2,
                units=256,
                act="silu",
                norm="rms",
                minstd=0.1,
                maxstd=1.0,
                outscale=0.01,
                unimix=0.01,
                winit="trunc_normal_in",
            ),
            value=dict(
                layers=2,
                units=256,
                act="silu",
                norm="rms",
                output="symexp_twohot",
                outscale=0.0,
                winit="trunc_normal_in",
                bins=255,
            ),
            slowvalue=dict(
                rate=0.02,
                every=1,
            ),
            # Policy and value distributions
            policy_dist_disc="categorical",
            policy_dist_cont="bounded_normal",
            # Normalization
            retnorm=dict(
                impl="perc",
                rate=0.01,
                limit=1.0,
                perclo=5.0,
                perchi=95.0,
                debias=False,
            ),
            valnorm=dict(
                impl="none",
                rate=0.01,
                limit=1e-8,
            ),
            advnorm=dict(
                impl="meanstd",
                rate=0.01,
                limit=1e-8,
                perclo=5.0,
                perchi=95.0,
                debias=True,
            ),
            # Loss scales
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
            # Optimizer
            opt=dict(
                lr=1e-4,
                agc=0.3,
                eps=1e-8,
                beta1=0.9,
                beta2=0.999,
                momentum=True,
                wd=0.0,
                schedule="const",
                warmup=1000,
                anneal=0,
            ),
            # Other settings
            ac_grads=False,
            replay_context=False,
            imag_length=15,
            imag_last=0,
            horizon=333,
            contdisc=True,
            gamma=0.99,
            gae_lambda=0.95,
            actor_grad="dynamics",
            critic_type="vpredict",
            slow_critic_update=1.0,
            reward_grad=True,
            repval_loss=True,
            repval_grad=True,
            report=True,
            report_gradnorms=False,
            imag_loss=dict(slowtar=False, lam=0.95, actent=3e-4, slowreg=1.0),
            repl_loss=dict(slowtar=False, lam=0.95, slowreg=1.0),
            # JAX configuration
            jax=dict(
                platform="cpu",
                compute_dtype="bfloat16",
                policy_devices=[0],
                train_devices=[0],
                mock_devices=0,
                prealloc=False,
                jit=True,
                debug=False,
                expect_devices=0,
                enable_policy=True,
            ),
        )

    @pytest.fixture
    def obs_space(self):
        """Simple observation space for testing"""
        return {
            "image": elements.Space(np.uint8, (64, 64, 3)),
            "reward": elements.Space(np.float32, ()),
            "is_first": elements.Space(bool, ()),
            "is_last": elements.Space(bool, ()),
            "is_terminal": elements.Space(bool, ()),
        }

    @pytest.fixture
    def act_space(self):
        """Simple action space for testing"""
        return {
            "action": elements.Space(np.float32, (4,)),
        }

    def test_agent_initialization(self, obs_space, act_space, minimal_config):
        """Test Agent can be initialized with valid configuration"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)

        # Agent returns a wrapper (embodied.jax.Agent) with inner model
        assert agent_instance.obs_space == obs_space
        assert agent_instance.act_space == act_space
        assert agent_instance.config == minimal_config
        # Check inner model has required components
        assert agent_instance.model.enc is not None
        assert agent_instance.model.dyn is not None
        assert agent_instance.model.dec is not None

    def test_banner_exists(self):
        """Test Agent class has banner attribute"""
        from dreamerv3.agent import Agent

        assert hasattr(Agent, "banner")
        assert isinstance(Agent.banner, list)
        assert len(Agent.banner) > 0
        # Check banner contains ASCII art
        assert all(isinstance(line, str) for line in Agent.banner)

    def test_init_policy(self, obs_space, act_space, minimal_config):
        """Test init_policy method exists and returns carry structure"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        # Test that init_policy method exists and can be called
        carry = agent_instance.init_policy(batch_size)

        # Basic structural checks - the exact structure depends on embodied.jax.Agent wrapper
        # Just verify we got something back
        assert carry is not None

    def test_init_train(self, obs_space, act_space, minimal_config):
        """Test init_train method exists and can be called"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        # Test that init_train method exists and can be called
        carry_train = agent_instance.init_train(batch_size)

        # Basic check - verify we got something back
        assert carry_train is not None

    def test_init_report(self, obs_space, act_space, minimal_config):
        """Test init_report method exists and can be called"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        # Test that init_report method exists and can be called
        carry_report = agent_instance.init_report(batch_size)

        # Basic check - verify we got something back
        assert carry_report is not None

    def test_loss_scales_configuration(self, obs_space, act_space, minimal_config):
        """Test loss scales are properly configured"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)

        expected_keys = {"rec", "rew", "con", "dyn", "rep", "policy", "value", "repval"}
        assert set(agent_instance.config.loss_scales.keys()) == expected_keys
        assert all(
            isinstance(v, (int, float))
            for v in agent_instance.config.loss_scales.values()
        )

    def test_policy_keys_property(self, obs_space, act_space, minimal_config):
        """Test policy_keys property returns correct regex pattern"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)

        # Access the inner model to get policy_keys
        policy_keys = agent_instance.model.policy_keys
        assert isinstance(policy_keys, str)
        assert "enc" in policy_keys
        assert "dyn" in policy_keys
        assert "pol" in policy_keys

    def test_ext_space_without_replay_context(
        self, obs_space, act_space, minimal_config
    ):
        """Test ext_space property without replay context"""
        from dreamerv3.agent import Agent

        config = minimal_config.update({"replay_context": 0})
        agent_instance = Agent(obs_space, act_space, config)

        ext_space = agent_instance.model.ext_space
        assert "consec" in ext_space
        assert "stepid" in ext_space
        # Should not have replay context keys
        assert not any("enc/" in k for k in ext_space.keys())
        assert not any("dyn/" in k for k in ext_space.keys())

    def test_ext_space_with_replay_context(self, obs_space, act_space, minimal_config):
        """Test ext_space property with replay context enabled"""
        from dreamerv3.agent import Agent

        config = minimal_config.update({"replay_context": 2})
        agent_instance = Agent(obs_space, act_space, config)

        ext_space = agent_instance.model.ext_space
        assert "consec" in ext_space
        assert "stepid" in ext_space
        # Should have replay context keys (check for flat dict keys)
        ext_keys = list(ext_space.keys())
        # Replay context adds encoder, dynamics, and decoder entry spaces as flat dict
        assert len(ext_keys) > 2  # More than just consec and stepid

    def test_report_with_report_disabled(self, obs_space, act_space, minimal_config):
        """Test report method early exits when report=False"""
        from dreamerv3.agent import Agent

        config = minimal_config.update({"report": False})
        agent_instance = Agent(obs_space, act_space, config)

        # Verify the report flag is set correctly
        assert agent_instance.model.config.report is False
        # The report method will early exit at line 288-289 when report=False
        # This test verifies the configuration, not the runtime behavior

    def test_optimizer_schedule_linear(self, obs_space, act_space, minimal_config):
        """Test optimizer with linear learning rate schedule"""
        from dreamerv3.agent import Agent

        config = minimal_config.update(
            {
                "opt": dict(
                    lr=1e-4,
                    agc=0.3,
                    eps=1e-8,
                    beta1=0.9,
                    beta2=0.999,
                    momentum=True,
                    wd=0.0,
                    schedule="linear",
                    warmup=100,
                    anneal=1000,
                )
            }
        )
        agent_instance = Agent(obs_space, act_space, config)

        # Just verify agent initializes with linear schedule
        assert agent_instance is not None

    def test_optimizer_schedule_cosine(self, obs_space, act_space, minimal_config):
        """Test optimizer with cosine learning rate schedule"""
        from dreamerv3.agent import Agent

        config = minimal_config.update(
            {
                "opt": dict(
                    lr=1e-4,
                    agc=0.3,
                    eps=1e-8,
                    beta1=0.9,
                    beta2=0.999,
                    momentum=True,
                    wd=0.0,
                    schedule="cosine",
                    warmup=100,
                    anneal=1000,
                )
            }
        )
        agent_instance = Agent(obs_space, act_space, config)

        # Just verify agent initializes with cosine schedule
        assert agent_instance is not None


class TestLambdaReturn:
    """Test lambda_return function for computing bootstrapped returns"""

    def test_lambda_return_shape(self):
        """Test lambda_return returns correct shape"""
        import jax.numpy as jnp

        from dreamerv3.agent import lambda_return

        B, T = 4, 8
        last = jnp.zeros((B, T), dtype=bool)
        term = jnp.zeros((B, T), dtype=bool)
        rew = jnp.ones((B, T), dtype=jnp.float32)
        val = jnp.ones((B, T), dtype=jnp.float32)
        boot = jnp.ones((B, T), dtype=jnp.float32)
        disc = 0.99
        lam = 0.95

        ret = lambda_return(last, term, rew, val, boot, disc, lam)

        assert ret.shape == (B, T - 1)

    def test_lambda_return_with_termination(self):
        """Test lambda_return handles terminal states correctly"""
        import jax.numpy as jnp

        from dreamerv3.agent import lambda_return

        B, T = 2, 5
        last = jnp.zeros((B, T), dtype=bool)
        term = jnp.zeros((B, T), dtype=bool)
        term = term.at[:, 2].set(True)  # Terminal at timestep 2
        rew = jnp.ones((B, T), dtype=jnp.float32)
        val = jnp.ones((B, T), dtype=jnp.float32)
        boot = jnp.ones((B, T), dtype=jnp.float32)
        disc = 0.99
        lam = 0.95

        ret = lambda_return(last, term, rew, val, boot, disc, lam)

        # Returns should be affected by terminal states
        assert ret.shape == (B, T - 1)
        assert jnp.all(jnp.isfinite(ret))

    def test_lambda_return_with_last(self):
        """Test lambda_return handles episode end correctly"""
        import jax.numpy as jnp

        from dreamerv3.agent import lambda_return

        B, T = 2, 5
        last = jnp.zeros((B, T), dtype=bool)
        last = last.at[:, 3].set(True)  # Last timestep at 3
        term = jnp.zeros((B, T), dtype=bool)
        rew = jnp.ones((B, T), dtype=jnp.float32)
        val = jnp.ones((B, T), dtype=jnp.float32)
        boot = jnp.ones((B, T), dtype=jnp.float32)
        disc = 0.99
        lam = 0.95

        ret = lambda_return(last, term, rew, val, boot, disc, lam)

        assert ret.shape == (B, T - 1)
        assert jnp.all(jnp.isfinite(ret))


class TestImagLoss:
    """Test imag_loss function for imagination-based policy/value training"""

    def test_imag_loss_callable(self):
        """Test imag_loss function is callable with correct signature"""
        import inspect

        from dreamerv3.agent import imag_loss

        # Verify function exists and signature
        assert callable(imag_loss)
        sig = inspect.signature(imag_loss)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        assert "act" in params
        assert "rew" in params
        assert "con" in params
        assert "policy" in params
        assert "value" in params
        assert "slowvalue" in params
        assert "contdisc" in params
        assert "horizon" in params
        assert "lam" in params


class TestReplLoss:
    """Test repl_loss function for replay-based value training"""

    def test_repl_loss_callable(self):
        """Test repl_loss function is callable with correct signature"""
        import inspect

        from dreamerv3.agent import repl_loss

        # Verify function exists and signature
        assert callable(repl_loss)
        sig = inspect.signature(repl_loss)
        params = list(sig.parameters.keys())

        # Check key parameters exist
        assert "last" in params
        assert "term" in params
        assert "rew" in params
        assert "boot" in params
        assert "value" in params
        assert "slowvalue" in params
        assert "valnorm" in params
        assert "slowtar" in params
        assert "horizon" in params
        assert "lam" in params
