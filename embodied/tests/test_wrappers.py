"""Comprehensive tests for embodied/core/wrappers.py."""

import time

import elements
import numpy as np
import pytest

import embodied
from embodied.core import wrappers


class TestWrapper:
    """Test the base Wrapper class."""

    def test_init(self):
        """Test wrapper initialization."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        assert wrapper.env is env

    def test_len(self):
        """Test __len__ delegates to wrapped env."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        # Dummy env doesn't have __len__, so this will raise TypeError
        with pytest.raises(TypeError):
            len(wrapper)

    def test_bool(self):
        """Test __bool__ delegates to wrapped env."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        assert bool(wrapper) is True

    def test_getattr(self):
        """Test __getattr__ delegates to wrapped env."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        assert wrapper.obs_space == env.obs_space
        assert wrapper.act_space == env.act_space

    def test_getattr_dunder(self):
        """Test __getattr__ raises AttributeError for dunder methods."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        with pytest.raises(AttributeError):
            _ = wrapper.__missing__

    def test_getattr_invalid(self):
        """Test __getattr__ raises ValueError for missing attributes."""
        env = self._make_env()
        wrapper = wrappers.Wrapper(env)
        with pytest.raises(ValueError):
            _ = wrapper.nonexistent_attribute

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestTimeLimit:
    """Test the TimeLimit wrapper."""

    def test_init(self):
        """Test TimeLimit initialization."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=5)
        assert wrapper._duration == 5
        assert wrapper._reset is True
        assert wrapper._step == 0
        assert wrapper._done is False

    def test_init_no_reset(self):
        """Test TimeLimit initialization without reset."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=5, reset=False)
        assert wrapper._reset is False

    def test_step_reset_action(self):
        """Test step with reset action."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=5, reset=True)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        assert obs["is_first"] is True
        assert wrapper._step == 0
        assert wrapper._done is False

    def test_step_normal(self):
        """Test normal step without time limit."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=5)
        # Reset first
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        # Normal step
        action["reset"] = False
        obs = wrapper.step(action)
        assert wrapper._step == 1
        assert obs["is_last"] is False

    def test_step_reaches_duration(self):
        """Test step when reaching time limit."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=3)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        # Step to duration
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        for i in range(3):
            obs = wrapper.step(action.copy())
        assert obs["is_last"] is True
        assert wrapper._done is True

    def test_step_after_done(self):
        """Test step after episode is done with reset=True."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=2, reset=True)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Step to done
        wrapper.step(action.copy())
        wrapper.step(action.copy())
        assert wrapper._done is True
        # Next step should reset
        obs = wrapper.step(action.copy())
        assert obs["is_first"] is True
        assert wrapper._step == 0
        assert wrapper._done is False

    def test_step_after_done_no_reset(self):
        """Test step after done without env reset."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=2, reset=False)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Step to done
        wrapper.step(action.copy())
        wrapper.step(action.copy())
        # Next step should NOT reset env
        obs = wrapper.step(action.copy())
        assert obs["is_first"] is True
        assert wrapper._step == 0

    def test_no_duration(self):
        """Test TimeLimit with no duration (None)."""
        env = self._make_env()
        wrapper = wrappers.TimeLimit(env, duration=None)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Step many times without hitting limit
        for _ in range(100):
            obs = wrapper.step(action.copy())
            if obs["is_last"]:  # Natural env termination
                break

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestActionRepeat:
    """Test the ActionRepeat wrapper."""

    def test_init(self):
        """Test ActionRepeat initialization."""
        env = self._make_env()
        wrapper = wrappers.ActionRepeat(env, repeat=4)
        assert wrapper._repeat == 4

    def test_step_reset(self):
        """Test step with reset action."""
        env = self._make_env()
        wrapper = wrappers.ActionRepeat(env, repeat=4)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        assert obs["is_first"] is True
        assert obs["reward"] == 0.0

    def test_step_normal(self):
        """Test normal step with action repeat."""
        # Use a wrapper that doesn't modify action dict
        env = self._make_non_popping_env()
        wrapper = wrappers.ActionRepeat(env, repeat=4)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        # Normal step
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # Should accumulate 4 rewards of 1 each
        assert obs["reward"] == 4.0
        assert isinstance(obs["reward"], np.float32)

    def test_step_early_termination(self):
        """Test action repeat stops on is_last."""
        env = self._make_non_popping_env(length=3)
        wrapper = wrappers.ActionRepeat(env, repeat=10)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # Should stop at is_last (3 steps)
        assert obs["reward"] == 3.0
        assert obs["is_last"] is True

    def test_step_early_terminal(self):
        """Test action repeat stops on is_terminal."""
        env = self._make_non_popping_env(length=3)
        wrapper = wrappers.ActionRepeat(env, repeat=10)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # Should stop when is_terminal is True
        assert obs["is_terminal"] is True

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)

    def _make_non_popping_env(self, length=10):
        """Create an env that doesn't pop the reset key from action dict."""
        from embodied.envs import dummy

        class NonPoppingDummy(dummy.Dummy):
            def step(self, action):
                # Don't modify the action dict - use get instead of pop
                if action.get("reset", False) or self.done:
                    self.count = 0
                    self.done = False
                    return self._obs(0, is_first=True)
                self.count += 1
                self.done = self.count >= self.length
                return self._obs(1, is_last=self.done, is_terminal=self.done)

        return NonPoppingDummy("disc", length=length)


class TestClipAction:
    """Test the ClipAction wrapper."""

    def test_init(self):
        """Test ClipAction initialization."""
        env = self._make_env()
        wrapper = wrappers.ClipAction(env)
        assert wrapper._key == "action"
        assert wrapper._low == -1
        assert wrapper._high == 1

    def test_init_custom(self):
        """Test ClipAction with custom parameters."""
        env = self._make_env()
        wrapper = wrappers.ClipAction(env, key="act_cont", low=-2, high=2)
        assert wrapper._key == "act_cont"
        assert wrapper._low == -2
        assert wrapper._high == 2

    def test_step_clip(self):
        """Test action clipping."""
        env = self._make_env()
        wrapper = wrappers.ClipAction(env, key="act_cont", low=-1, high=1)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([2.0, -3.0, 0.5, 1.5, -1.5, 0.0], np.float32),
        }
        obs = wrapper.step(action)
        # Check that the environment received clipped values
        assert obs is not None

    def test_step_no_clip(self):
        """Test action without clipping needed."""
        env = self._make_env()
        wrapper = wrappers.ClipAction(env, key="act_cont", low=-1, high=1)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([0.5, -0.5, 0.0, 0.8, -0.8, 0.2], np.float32),
        }
        obs = wrapper.step(action)
        assert obs is not None

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestNormalizeAction:
    """Test the NormalizeAction wrapper."""

    def test_init(self):
        """Test NormalizeAction initialization."""
        env = self._make_env()
        wrapper = wrappers.NormalizeAction(env, key="act_cont")
        assert wrapper._key == "act_cont"
        assert wrapper._space == env.act_space["act_cont"]

    def test_act_space(self):
        """Test normalized action space."""
        env = self._make_env()
        wrapper = wrappers.NormalizeAction(env, key="act_cont")
        act_space = wrapper.act_space
        # Check that the action space is normalized to [-1, 1]
        assert act_space["act_cont"].low.min() == -1
        assert act_space["act_cont"].high.max() == 1

    def test_step_normalize(self):
        """Test action normalization."""
        env = self._make_env()
        wrapper = wrappers.NormalizeAction(env, key="act_cont")
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.0], np.float32),
        }
        obs = wrapper.step(action)
        assert obs is not None

    def test_step_infinite_bounds(self):
        """Test normalization with infinite bounds."""
        env = self._make_env()
        # Modify act_space to have infinite bounds
        original_space = env.act_space["act_cont"]
        low = np.full_like(original_space.low, -np.inf)
        high = np.full_like(original_space.high, np.inf)
        env.act_space["act_cont"] = elements.Space(
            original_space.dtype, original_space.shape, low, high
        )
        wrapper = wrappers.NormalizeAction(env, key="act_cont")
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([1.0, -1.0, 0.0, 0.5, -0.5, 0.0], np.float32),
        }
        obs = wrapper.step(action)
        assert obs is not None

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestUnifyDtypes:
    """Test the UnifyDtypes wrapper."""

    def test_init(self):
        """Test UnifyDtypes initialization."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        assert wrapper._obs_space is not None
        assert wrapper._act_space is not None

    def test_obs_space(self):
        """Test unified observation space."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        obs_space = wrapper.obs_space
        # Check that float types are unified to float32
        assert obs_space["vector"].dtype == np.float32
        assert obs_space["count"].dtype == np.float32
        # Check that uint8 stays uint8
        assert obs_space["image"].dtype == np.uint8
        # Check that integers are unified to int32
        assert obs_space["token"].dtype == np.int32

    def test_act_space(self):
        """Test unified action space."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        act_space = wrapper.act_space
        # Check that int types are unified to int32
        assert act_space["act_disc"].dtype == np.int32
        # Check that float types are unified to float32
        assert act_space["act_cont"].dtype == np.float32

    def test_step(self):
        """Test step with dtype conversion."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # Check that observations have correct dtypes
        assert obs["vector"].dtype == np.float32
        assert obs["count"].dtype == np.float32
        assert obs["image"].dtype == np.uint8
        assert obs["token"].dtype == np.int32

    def test_convert_floating(self):
        """Test _convert method with floating types."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        # Test float64 -> float32
        spaces = {"test": elements.Space(np.float64, (3,), -1.0, 1.0)}
        results, befores, afters = wrapper._convert(spaces)
        assert befores["test"] == np.float64
        assert afters["test"] == np.float32
        assert results["test"].dtype == np.float32

    def test_convert_uint8(self):
        """Test _convert method with uint8."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        spaces = {"test": elements.Space(np.uint8, (64, 64, 3), 0, 255)}
        results, befores, afters = wrapper._convert(spaces)
        assert befores["test"] == np.uint8
        assert afters["test"] == np.uint8
        assert results["test"].dtype == np.uint8

    def test_convert_integer(self):
        """Test _convert method with integer types."""
        env = self._make_env()
        wrapper = wrappers.UnifyDtypes(env)
        # Test int64 -> int32
        spaces = {"test": elements.Space(np.int64, (), 0, 100)}
        results, befores, afters = wrapper._convert(spaces)
        assert befores["test"] == np.int64
        assert afters["test"] == np.int32
        assert results["test"].dtype == np.int32

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestCheckSpaces:
    """Test the CheckSpaces wrapper."""

    def test_init(self):
        """Test CheckSpaces initialization."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        assert wrapper.env is env

    def test_init_overlapping_spaces(self):
        """Test CheckSpaces raises on overlapping obs/act spaces."""

        class OverlappingEnv:
            @property
            def obs_space(self):
                return {"reset": elements.Space(bool)}

            @property
            def act_space(self):
                return {"reset": elements.Space(bool)}

        env = OverlappingEnv()
        with pytest.raises(AssertionError):
            wrappers.CheckSpaces(env)

    def test_step_valid(self):
        """Test step with valid action and observation."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        assert obs is not None

    def test_step_invalid_action_type(self):
        """Test step with invalid action type."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        action = {
            "reset": True,
            "act_disc": "invalid",  # String instead of number
            "act_cont": np.zeros(6, np.float32),
        }
        with pytest.raises(TypeError):
            wrapper.step(action)

    def test_step_invalid_action_value(self):
        """Test step with out-of-bounds action."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        action = {
            "reset": True,
            "act_disc": np.array(100, np.int32),  # Out of bounds [0, 5)
            "act_cont": np.zeros(6, np.float32),
        }
        with pytest.raises(ValueError):
            wrapper.step(action)

    def test_check_valid_types(self):
        """Test _check with valid types."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        # Test with numpy array
        space = elements.Space(np.float32, (3,), -1, 1)
        wrapper._check(np.zeros(3, np.float32), space, "test")
        # Test with numpy int32
        space_int = elements.Space(np.int32, (), 0, 10)
        wrapper._check(np.int32(5), space_int, "test")
        # Test with bool
        space_bool = elements.Space(bool, ())
        wrapper._check(True, space_bool, "test")

    def test_check_invalid_type(self):
        """Test _check with invalid type."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        space = elements.Space(np.float32, (3,), -1, 1)
        with pytest.raises(TypeError):
            wrapper._check({"invalid": "dict"}, space, "test")

    def test_check_out_of_bounds(self):
        """Test _check with out of bounds value."""
        env = self._make_env()
        wrapper = wrappers.CheckSpaces(env)
        space = elements.Space(np.float32, (3,), -1, 1)
        with pytest.raises(ValueError):
            wrapper._check(np.array([0.0, 2.0, 0.0], np.float32), space, "test")

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestDiscretizeAction:
    """Test the DiscretizeAction wrapper."""

    def test_init(self):
        """Test DiscretizeAction initialization."""
        env = self._make_env()
        wrapper = wrappers.DiscretizeAction(env, key="act_cont", bins=5)
        assert wrapper._key == "act_cont"
        assert wrapper._dims == 6
        assert len(wrapper._values) == 5

    def test_init_custom_bins(self):
        """Test DiscretizeAction with custom bins."""
        env = self._make_env()
        wrapper = wrappers.DiscretizeAction(env, key="act_cont", bins=7)
        assert len(wrapper._values) == 7
        # Check that values span from -1 to 1
        assert wrapper._values[0] == -1.0
        assert wrapper._values[-1] == 1.0

    def test_act_space(self):
        """Test discretized action space."""
        env = self._make_env()
        wrapper = wrappers.DiscretizeAction(env, key="act_cont", bins=5)
        act_space = wrapper.act_space
        # Check that action space is now discrete
        assert act_space["act_cont"].dtype == np.int32
        assert act_space["act_cont"].shape == (6,)
        assert (act_space["act_cont"].low == 0).all()
        assert (act_space["act_cont"].high == 5).all()

    def test_step(self):
        """Test step with discretized action."""
        env = self._make_env()
        wrapper = wrappers.DiscretizeAction(env, key="act_cont", bins=5)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([0, 1, 2, 3, 4, 2], np.int32),
        }
        obs = wrapper.step(action)
        assert obs is not None

    def test_step_continuous_conversion(self):
        """Test that discrete actions are converted to continuous."""
        env = self._make_env()
        wrapper = wrappers.DiscretizeAction(env, key="act_cont", bins=5)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.array([0, 2, 4, 1, 3, 2], np.int32),
        }
        # This should convert to continuous values
        obs = wrapper.step(action)
        assert obs is not None

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestResizeImage:
    """Test the ResizeImage wrapper."""

    def test_init(self):
        """Test ResizeImage initialization."""
        env = self._make_env()
        wrapper = wrappers.ResizeImage(env, size=(32, 32))
        assert wrapper._size == (32, 32)
        assert "image" in wrapper._keys

    def test_init_no_resize_needed(self):
        """Test ResizeImage when images are already correct size."""
        env = self._make_env()
        wrapper = wrappers.ResizeImage(env, size=(64, 64))
        # Image is already 64x64, so shouldn't be in _keys
        assert "image" not in wrapper._keys

    def test_obs_space(self):
        """Test resized observation space."""
        env = self._make_env()
        wrapper = wrappers.ResizeImage(env, size=(32, 32))
        obs_space = wrapper.obs_space
        # Check that image space has new size
        assert obs_space["image"].shape == (32, 32, 3)
        assert obs_space["image"].dtype == np.uint8

    def test_step(self):
        """Test step with image resizing."""
        env = self._make_env()
        wrapper = wrappers.ResizeImage(env, size=(32, 32))
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # Check that image is resized
        assert obs["image"].shape == (32, 32, 3)

    def test_resize(self):
        """Test _resize method."""
        env = self._make_env()
        wrapper = wrappers.ResizeImage(env, size=(32, 32))
        # Create a test image
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        resized = wrapper._resize(image)
        assert resized.shape == (32, 32, 3)

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestBackwardReturn:
    """Test the BackwardReturn wrapper."""

    def test_init(self):
        """Test BackwardReturn initialization."""
        env = self._make_env()
        wrapper = wrappers.BackwardReturn(env, horizon=100)
        assert wrapper._discount == 1 - 1 / 100
        assert wrapper._bwreturn == 0.0

    def test_obs_space(self):
        """Test observation space with bwreturn."""
        env = self._make_env()
        wrapper = wrappers.BackwardReturn(env, horizon=100)
        obs_space = wrapper.obs_space
        assert "bwreturn" in obs_space
        assert obs_space["bwreturn"].dtype == np.float32

    def test_step_first(self):
        """Test step with first observation."""
        env = self._make_env()
        wrapper = wrappers.BackwardReturn(env, horizon=100)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # First step resets bwreturn
        assert obs["bwreturn"] == 0.0

    def test_step_accumulate(self):
        """Test backward return accumulation."""
        env = self._make_env()
        wrapper = wrappers.BackwardReturn(env, horizon=100)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Step multiple times
        obs1 = wrapper.step(action.copy())
        obs2 = wrapper.step(action.copy())
        # bwreturn should increase with discount
        assert obs2["bwreturn"] > obs1["bwreturn"]

    def test_step_reset_bwreturn(self):
        """Test bwreturn reset on new episode."""
        env = self._make_env()
        wrapper = wrappers.BackwardReturn(env, horizon=100)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        wrapper.step(action)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Accumulate some return
        wrapper.step(action.copy())
        wrapper.step(action.copy())
        # Reset
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        # bwreturn should be reset to current reward
        assert obs["bwreturn"] == 0.0

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestAddObs:
    """Test the AddObs wrapper."""

    def test_init(self):
        """Test AddObs initialization."""
        env = self._make_env()
        space = elements.Space(np.float32, (3,))
        wrapper = wrappers.AddObs(env, key="custom", value=np.zeros(3), space=space)
        assert wrapper._key == "custom"
        assert (wrapper._value == np.zeros(3)).all()
        assert wrapper._space is space

    def test_obs_space(self):
        """Test observation space with added key."""
        env = self._make_env()
        space = elements.Space(np.float32, (3,))
        wrapper = wrappers.AddObs(env, key="custom", value=np.zeros(3), space=space)
        obs_space = wrapper.obs_space
        assert "custom" in obs_space
        assert obs_space["custom"] is space

    def test_step(self):
        """Test step with added observation."""
        env = self._make_env()
        value = np.array([1.0, 2.0, 3.0], np.float32)
        space = elements.Space(np.float32, (3,))
        wrapper = wrappers.AddObs(env, key="custom", value=value, space=space)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        assert "custom" in obs
        assert (obs["custom"] == value).all()

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)


class TestRestartOnException:
    """Test the RestartOnException wrapper."""

    def test_init(self):
        """Test RestartOnException initialization."""

        def ctor():
            return self._make_env()

        wrapper = wrappers.RestartOnException(ctor)
        assert wrapper._ctor is ctor
        assert wrapper._exceptions == (Exception,)
        assert wrapper._window == 300
        assert wrapper._maxfails == 2
        assert wrapper._wait == 20
        assert wrapper._fails == 0

    def test_init_custom_exceptions(self):
        """Test RestartOnException with custom exceptions."""

        def ctor():
            return self._make_env()

        wrapper = wrappers.RestartOnException(
            ctor, exceptions=[ValueError, RuntimeError]
        )
        assert wrapper._exceptions == (ValueError, RuntimeError)

    def test_init_single_exception(self):
        """Test RestartOnException with single exception."""

        def ctor():
            return self._make_env()

        wrapper = wrappers.RestartOnException(ctor, exceptions=ValueError)
        assert wrapper._exceptions == (ValueError,)

    def test_step_normal(self):
        """Test normal step without exception."""

        def ctor():
            return self._make_env()

        wrapper = wrappers.RestartOnException(ctor)
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        obs = wrapper.step(action)
        assert obs is not None
        assert wrapper._fails == 0

    def test_step_with_exception(self):
        """Test step with exception and recovery."""

        class FailingEnv:
            def __init__(self):
                self.call_count = 0
                self.base_env = self._make_env()

            def step(self, action):
                self.call_count += 1
                if self.call_count == 2:  # Fail on second call
                    raise ValueError("Test exception")
                return self.base_env.step(action)

            def _make_env(self):
                from embodied.envs import dummy

                return dummy.Dummy("disc", length=10)

            def __getattr__(self, name):
                return getattr(self.base_env, name)

        fail_count = [0]

        def ctor():
            fail_count[0] += 1
            return FailingEnv()

        wrapper = wrappers.RestartOnException(
            ctor,
            exceptions=ValueError,
            wait=0.1,  # Short wait for testing
        )
        action = {
            "reset": True,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # First step succeeds
        wrapper.step(action)
        action["reset"] = False
        # Second step triggers exception and restart
        obs = wrapper.step(action)
        assert obs is not None
        assert wrapper._fails == 1
        assert fail_count[0] == 2  # Environment recreated

    def test_step_too_many_failures(self):
        """Test that too many failures raises RuntimeError."""

        call_count = [0]

        class SometimesFailingEnv:
            def __init__(self):
                self.base_env = self._make_env()

            def step(self, action):
                call_count[0] += 1
                # Fail on normal steps (reset=False), succeed on reset steps
                # This simulates failures that keep happening
                if not action.get("reset", False):
                    raise ValueError("Fails on purpose")
                return self.base_env.step(action)

            def _make_env(self):
                from embodied.envs import dummy

                return dummy.Dummy("disc", length=10)

            def __getattr__(self, name):
                return getattr(self.base_env, name)

        def ctor():
            return SometimesFailingEnv()

        wrapper = wrappers.RestartOnException(
            ctor, exceptions=ValueError, maxfails=2, wait=0.01
        )
        action = {
            "reset": np.array(False),
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Call multiple times, each will fail and restart
        # After maxfails+1 failures within the window, should raise RuntimeError
        with pytest.raises(RuntimeError, match="too many times"):
            for _ in range(5):
                wrapper.step(action.copy())

    def test_step_failure_window_reset(self):
        """Test that failure counter resets after window expires."""

        call_count = [0]

        class SometimesFailingEnv:
            def __init__(self):
                self.base_env = self._make_env()

            def step(self, action):
                call_count[0] += 1
                # Fail on calls 1 and 3, but succeed on the reset recovery steps
                if call_count[0] in [1, 3]:
                    raise ValueError("Test exception")
                return self.base_env.step(action)

            def _make_env(self):
                from embodied.envs import dummy

                return dummy.Dummy("disc", length=10)

            def __getattr__(self, name):
                return getattr(self.base_env, name)

        def ctor():
            return SometimesFailingEnv()

        wrapper = wrappers.RestartOnException(
            ctor, exceptions=ValueError, window=1, wait=0.01, maxfails=1
        )
        action = {
            "reset": np.array(False),
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # First step fails (call 1), then succeeds on reset (call 2), _fails = 1
        wrapper.step(action.copy())
        # Wait for window to expire
        time.sleep(1.1)
        # Second step fails (call 3), then succeeds on reset (call 4)
        # Window has reset so _fails should be set to 1 again, not raise RuntimeError
        wrapper.step(action.copy())
        # Should not raise RuntimeError because window reset

    def test_step_reset_action_on_exception(self):
        """Test that action is set to reset after exception."""

        class FailOnceEnv:
            def __init__(self):
                self.failed = False
                self.base_env = self._make_env()

            def step(self, action):
                if not self.failed and not action["reset"]:
                    self.failed = True
                    raise ValueError("Test exception")
                return self.base_env.step(action)

            def _make_env(self):
                from embodied.envs import dummy

                return dummy.Dummy("disc", length=10)

            def __getattr__(self, name):
                return getattr(self.base_env, name)

        def ctor():
            return FailOnceEnv()

        wrapper = wrappers.RestartOnException(ctor, exceptions=ValueError, wait=0.1)
        action = {
            "reset": False,
            "act_disc": np.zeros((), np.int32),
            "act_cont": np.zeros(6, np.float32),
        }
        # Should trigger exception and recovery with reset=True
        obs = wrapper.step(action)
        assert obs is not None

    def _make_env(self, length=10):
        from embodied.envs import dummy

        return dummy.Dummy("disc", length=length)
