"""Small stdlib-curses application: task → settings → live training screen."""
from __future__ import annotations

import copy
import curses
import signal
import threading
import time
from .config import load_train_config
from .server_inputs import REWARD_SOURCES, choose_task, discover_tasks
from .models import MODELS
from .server_config import distributed_config
from .server_pipeline import ServerRun, execution_plan, training_settings


def fields(raw):
    common = [("Steps", "training.train_steps", int), ("Global micro batch", "training.micro_batch_size", int),
        ("Gradient accumulation", "training.gradient_accumulation_steps", int),
        ("Checkpoint interval", "training.checkpoint_interval", int), ("Seed", "training.seed", int),
        ("Actor peak LR", "training.policy_peak_lr", float), ("Actor final LR", "training.policy_final_lr", float),
        ("Actor LR warmup", "training.actor_lr_warmup_steps", int),
        ("Resume checkpoint (empty = new run)", "training.resume_checkpoint", str),
        ("Model family", "model.family", tuple(MODELS)),
        ("Base checkpoint", "model.base_checkpoint", str),
        ("Normalization stats", "model.stats_key", str),
        ("Backbone", "model.backbone", ("frozen", "lora", "full")),
        ("Action head", "model.action_head", ("train", "frozen")),
        ("Proprio projector", "model.proprio_projector", ("train", "frozen")),
        ("Keep post-success actions", "data.include_post_success", (True, False)),
        ("TensorBoard", "logging.tensorboard", (True, False)),
        ("W&B", "logging.wandb.enabled", (False, True)),
        ("W&B mode", "logging.wandb.mode", ("online", "offline")),
        ("W&B project", "logging.wandb.project", str),
        ("W&B entity (empty = default)", "logging.wandb.entity", str)]
    if raw["model"]["backbone"] == "lora":
        common += [("LoRA rank", "model.lora.rank", int), ("LoRA alpha", "model.lora.alpha", int),
                   ("LoRA dropout", "model.lora.dropout", float)]
    if raw["training"]["method"] == "iql":
        common += [("Discount gamma", "reward.gamma", float)]
        if not (raw["reward"].get("source") == "final" and raw["reward"].get("fusion_mode") == "multiplicative"):
            common += [("Cumulative reward", "reward.accumulate_primitive_steps", (False, True))]
        common += [(key.replace("_", " "), f"iql.{key}",
                    ("adam", "adamw") if key.endswith("optimizer") else type(value))
                   for key, value in raw["iql"].items()]
    return common


def get_value(raw, path):
    for key in path.split("."):
        raw = raw[key]
    return raw


def set_value(raw, path, value):
    keys = path.split(".")
    for key in keys[:-1]:
        raw = raw.setdefault(key, {})
    raw[keys[-1]] = value


class ServerApp:
    def __init__(self, server):
        self.server = server
        project = load_train_config(server.training_config).raw["data"]["project_id"]
        self.tasks = discover_tasks(server.runs_root, project)
        self.task = choose_task(self.tasks, server.task_id) if server.task_id else None
        self.source = server.reward_source
        self.overrides = copy.deepcopy(server.overrides)
        self.distributed = server.distributed
        self.page = "settings" if self.task else "tasks"
        self.cursor, self.message = 0, ""
        self.run = None
        self.start_thread = None
        self.error = None
        self.stop_requested = False
        self.settings_key = None
        self.settings_error = None

    def _lines(self):
        if self.page == "tasks":
            return [f"{task.task_id} | {task.summary()['marked']}/{len(task.runs)} marked | " +
                    " ".join(f"{source}:{count}" for source, count in task.summary()["rewards"].items())
                    for task in self.tasks] or ["No runs. Copy complete PC run directories with global evaluations here."]
        key = (self.task.task_id, self.source, self.overrides)
        if key != self.settings_key:
            self.settings_error = None
            try:
                self.raw = training_settings(self.server, self.task, overrides=self.overrides, source=self.source)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                # Keep the controls accessible to choose BC or another source;
                # this fallback is display-only and can never launch training.
                self.settings_error = str(exc)
                self.raw = load_train_config(self.server.training_config, overrides=self.overrides,
                                            overrides_path=self.server.path).raw
                self.raw["reward"]["source"] = self.source
            self.settings_key = copy.deepcopy(key)
        raw = self.raw
        if self.settings_error:
            self.message = f"Cannot start: {self.settings_error} (change Method / Reward or repair the file)"
        self.options = fields(raw)
        eligible = self.task.selected(self.source if raw['training']['method'] == 'iql' else None)
        return [f"START TRAINING ({len(eligible)}/{len(self.task.runs)} eligible runs; all selected)",
                "Choose another task", f"Method: {raw['training']['method']}",
                f"Reward: {self.source}" + (" (not used by BC)" if raw['training']['method'] == "bc" else ""),
                f"GPUs: {','.join(map(str, self.distributed.gpu_ids))}"] + [
            f"{label}: {get_value(raw, path)}" for label, path, _ in self.options]

    @staticmethod
    def draw(screen, row, text, *, selected=False):
        height, width = screen.getmaxyx()
        if 0 <= row < height - 1:
            try:
                screen.addnstr(row, 1, str(text).replace("\t", " "), max(0, width - 3),
                               curses.A_REVERSE if selected else curses.A_NORMAL)
            except curses.error:
                pass

    def edit(self, screen, current):
        curses.echo()
        curses.curs_set(1)
        try:
            height, width = screen.getmaxyx()
            screen.move(height - 3, 1)
            screen.clrtoeol()
            screen.addnstr(f"New value (current: {current}): ", width - 3)
            screen.refresh()
            screen.timeout(-1)
            return screen.getstr().decode("utf-8").strip()
        finally:
            screen.timeout(200)
            curses.noecho()
            curses.curs_set(0)

    def activate(self, screen):
        if self.page == "tasks":
            if self.tasks:
                self.task = self.tasks[self.cursor]
                self.page, self.cursor = "settings", 0
            return
        if self.cursor == 0:
            if self.settings_error:
                raise ValueError(self.settings_error)
            execution_plan(self.server, self.task, self.raw, self.distributed, self.source)
            self.run = ServerRun(self.server, self.task, self.raw, self.distributed, self.source)
            self.page = "running"
            def start():
                try:
                    self.run.start()
                except Exception as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
            self.start_thread = threading.Thread(target=start, daemon=False)
            self.start_thread.start()
        elif self.cursor == 1:
            self.page, self.cursor = "tasks", 0
        elif self.cursor == 2:
            current = self.raw["training"]["method"]
            self.overrides["training"]["method"] = "bc" if current == "iql" else "iql"
        elif self.cursor == 3:
            self.source = REWARD_SOURCES[(REWARD_SOURCES.index(self.source) + 1) % len(REWARD_SOURCES)]
        elif self.cursor == 4:
            value = self.edit(screen, self.distributed.gpu_ids)
            ids = [int(item.strip()) for item in value.split(",")]
            self.distributed = distributed_config({**self.distributed.__dict__, "gpu_ids": ids})
        else:
            _, path, kind = self.options[self.cursor - 5]
            value = get_value(self.raw, path)
            if isinstance(kind, tuple):
                value = kind[(kind.index(value) + 1) % len(kind)]
            else:
                value = self.edit(screen, value)
                value = None if path in {"training.resume_checkpoint", "logging.wandb.entity"} and not value else kind(value)
            candidate = copy.deepcopy(self.overrides)
            set_value(candidate, path, value)
            training_settings(self.server, self.task, overrides=candidate, source=self.source)
            self.overrides = candidate

    def running_lines(self):
        if self.start_thread.is_alive():
            return ["Verifying transfer / preparing local manifests...", "No reward-model inference will run."]
        if self.error:
            return [self.error, "Press q to exit; inspect pipeline.json for details."]
        if self.stop_requested:
            self.run.cancel()
        code = self.run.poll()
        progress = self.run.progress()
        return [f"Status: {self.run.state['status']}", f"Run: {self.run.directory}",
                f"Step: {progress.get('step', 0)} / {self.raw['training']['train_steps']}",
                f"Speed: {progress.get('samples_per_second', 0):.2f} samples/s | "
                f"ETA: {progress.get('estimated_remaining_seconds', 0) / 3600:.2f} h",
                f"Data: {progress.get('data_wait_seconds', 0):.3f}s | Q/V: {progress.get('critic_seconds', 0):.3f}s | "
                f"Actor: {progress.get('actor_seconds', 0):.3f}s", "", *self.run.tail(), "",
                "q / Ctrl+C: request checkpoint and stop" if code is None else "q: exit"]

    def __call__(self, screen):
        curses.curs_set(0)
        screen.timeout(200)
        while True:
            screen.erase()
            self.draw(screen, 0, "VLA server | task -> configuration -> Enter on START | q to quit")
            try:
                lines = self.running_lines() if self.page == "running" else self._lines()
            except (ValueError, KeyError) as exc:
                self.message = str(exc)
                self.page, self.cursor = "tasks", 0
                lines = self._lines()
            height = max(1, screen.getmaxyx()[0] - 6)
            offset = 0 if self.page == "running" else max(0, self.cursor - height + 1)
            for index, line in enumerate(lines[offset:offset + height]):
                self.draw(screen, index + 2, line, selected=self.page != "running" and index + offset == self.cursor)
            self.draw(screen, screen.getmaxyx()[0] - 2, self.message)
            screen.refresh()
            try:
                key = screen.getch()
            except KeyboardInterrupt:
                key = ord("q")
            if self.stop_requested:
                key = ord("q")
            if key in (ord("q"), 27, 3):
                if self.page != "running":
                    return
                if self.start_thread.is_alive():
                    self.stop_requested = True
                    self.message = "Stop requested; finishing transfer verification before safe cancellation."
                elif self.error or self.run.poll() is not None:
                    return
                else:
                    self.run.cancel()
                    self.message = "Stopping at next complete actor update; saving ZeRO checkpoint..."
            elif self.page != "running":
                if key in (curses.KEY_UP, ord("k")):
                    self.cursor = max(0, self.cursor - 1)
                elif key in (curses.KEY_DOWN, ord("j")):
                    self.cursor = min(len(lines) - 1, self.cursor + 1)
                elif key in (10, 13, curses.KEY_ENTER):
                    try:
                        self.activate(screen)
                        self.message = ""
                    except (ValueError, TypeError, KeyError, OSError) as exc:
                        self.message = str(exc)


def run_tui(server):
    app = ServerApp(server)
    def stop(*_):
        app.stop_requested = True
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        curses.wrapper(app)
    finally:
        # Exceptions or resize failures must never orphan a torchrun group.
        if app.start_thread:
            app.start_thread.join()
        if app.run and app.run.process and app.run.poll() is None:
            app.run.cancel()
            while app.run.poll() is None:
                time.sleep(.2)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
