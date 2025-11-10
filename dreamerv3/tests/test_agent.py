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

        assert agent_instance.obs_space == obs_space
        assert agent_instance.act_space == act_space
        assert agent_instance.config == minimal_config
        assert agent_instance.enc is not None
        assert agent_instance.dyn is not None
        assert agent_instance.dec is not None

    def test_banner_exists(self):
        """Test Agent class has banner attribute"""
        from dreamerv3.agent import Agent

        assert hasattr(Agent, "banner")
        assert isinstance(Agent.banner, list)
        assert len(Agent.banner) > 0
        # Check banner contains ASCII art
        assert all(isinstance(line, str) for line in Agent.banner)

    def test_init_policy(self, obs_space, act_space, minimal_config):
        """Test init_policy returns proper carry structure"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        carry = agent_instance.init_policy(batch_size)

        # Should return tuple of (enc_carry, dyn_carry, dec_carry, prevact)
        assert isinstance(carry, tuple)
        assert len(carry) == 4

        enc_carry, dyn_carry, dec_carry, prevact = carry

        # Check prevact has correct shape
        assert "action" in prevact
        assert prevact["action"].shape == (batch_size, 4)

    def test_init_train(self, obs_space, act_space, minimal_config):
        """Test init_train returns same structure as init_policy"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        carry_policy = agent_instance.init_policy(batch_size)
        carry_train = agent_instance.init_train(batch_size)

        # Should have same structure
        assert len(carry_policy) == len(carry_train)

    def test_init_report(self, obs_space, act_space, minimal_config):
        """Test init_report returns same structure as init_policy"""
        from dreamerv3.agent import Agent

        agent_instance = Agent(obs_space, act_space, minimal_config)
        batch_size = 4

        carry_policy = agent_instance.init_policy(batch_size)
        carry_report = agent_instance.init_report(batch_size)

        # Should have same structure
        assert len(carry_policy) == len(carry_report)

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
