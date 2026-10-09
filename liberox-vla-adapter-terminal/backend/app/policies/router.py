"""One resident model at a time, always selected by verified policy identity."""
from __future__ import annotations

from .vla_adapter import VLAAdapterPolicyProvider
from .pi05 import Pi05PolicyProvider


class PolicyProvider:
    def __init__(self, runtime, eval_config, catalog):
        self.runtime, self.eval_config, self.catalog = runtime, eval_config, catalog
        self.provider = VLAAdapterPolicyProvider(runtime, eval_config, catalog)
        self.family = "vla_adapter"

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
        factories = {"vla_adapter": VLAAdapterPolicyProvider, "pi05": Pi05PolicyProvider}
        if entry.family not in factories:
            raise ValueError(f"Unsupported policy family: {entry.family}")
        if entry.family != self.family:
            self.unload()
            self.provider = factories[entry.family](self.runtime, self.eval_config, self.catalog)
            self.family = entry.family
        self.provider.load(open_loop_steps, policy_id, expected_content_sha256=expected_content_sha256)

    def seed(self, seed):
        if isinstance(self.provider, Pi05PolicyProvider):
            self.provider.seed(seed)

    def unload(self):
        self.provider.unload()

    def predict(self, *args, **kwargs):
        return self.provider.predict(*args, **kwargs)

    def process_action(self, action):
        return self.provider.process_action(action)

    def metadata(self):
        return self.provider.metadata()
