"""One resident model at a time, always selected by verified policy identity."""
from __future__ import annotations

from .registry import model_module, provider_factory


class PolicyProvider:
    def __init__(self, runtime, eval_config, catalog):
        self.runtime, self.eval_config, self.catalog = runtime, eval_config, catalog
        self.family = model_module("models").DEFAULT_MODEL["family"]
        self.provider = provider_factory(self.family)(runtime, eval_config, catalog)

    @property
    def loaded(self):
        return self.provider.loaded

    @property
    def current_policy_id(self):
        return self.provider.current_policy_id

    @property
    def current_policy_entry(self):
        return self.provider.current_policy_entry

    def load(self, open_loop_steps, policy_id="base", *, expected_content_sha256=None):
        self.catalog.refresh()
        entry = self.catalog.entry(policy_id)
        open_loop_steps = entry.resolve_open_loop_steps(open_loop_steps)
        if entry.family != self.family:
            self.unload()
            self.provider = provider_factory(entry.family)(self.runtime, self.eval_config, self.catalog)
            self.family = entry.family
        try:
            self.provider.load(open_loop_steps, policy_id, expected_content_sha256=expected_content_sha256)
            loaded = self.current_policy_entry
            if (not self.loaded or loaded is None or self.current_policy_id != policy_id
                    or (loaded.policy_id, loaded.family) != (policy_id, entry.family)
                    or loaded.content_sha256 != self.catalog.entry(policy_id).content_sha256
                    or (expected_content_sha256 is not None and loaded.content_sha256 != expected_content_sha256)):
                raise ValueError(f"Loaded model does not match the selected policy: {policy_id}")
        except BaseException:
            self.unload()
            raise

    def seed(self, seed):
        if hasattr(self.provider, "seed"):
            self.provider.seed(seed)

    def unload(self):
        self.provider.unload()

    def predict(self, *args, **kwargs):
        return self.provider.predict(*args, **kwargs)

    def process_action(self, action):
        return self.provider.process_action(action)

    def metadata(self):
        result = self.provider.metadata()
        adapter = model_module("base_models").family_adapter(self.family)
        entry = self.current_policy_entry
        if entry is not None:
            result = {**result, "model_family": entry.family, "variant_id": entry.settings["base_id"],
                      "policy_content_sha256": entry.content_sha256,
                      "observation": model_module("base_models").model_observation(entry.settings)}
        return {**result, "recording_action_codec": adapter.RECORDING_ACTION_CODEC}
