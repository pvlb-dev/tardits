import math
import time
import numpy as np
import torch
from tardits.modules.conf import MLFlowConfig

try:
    import mlflow
except ImportError:
    mlflow = None


class ConsoleLogger:
    def on_train_start(self, trainer):
        print(f"trainable parameters: {trainer.model.get_num_params()}")
        print("pre-heating CPU cores 🔥\nflooding RAM with floats 🌊\n")
        self.start_time = time.time()

    def on_eval(self, trainer, losses):
        print(
            f"🧠💦 step {trainer.step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    def on_step(self, trainer, loss):
        return

    def on_train_end(self, trainer, losses):
        elapsed = time.time() - self.start_time
        print(f"🏁🏁🏁\n⌛ training finished after {elapsed / 60:.2f} minutes!\n")
        print(f"""**************************************
results:
    train loss: {losses["train"]:.4f}
    val loss:   {losses["val"]:.4f}
settings:
    epochs: {trainer.config.max_iters}
    batch-size: {trainer.config.batch_size}
    learning-rate: {trainer.config.learning_rate}
**************************************""")


class MLFlowLogger:
    def __init__(self, config: MLFlowConfig):
        self.config = config
        self.frames_data = []  # Für SAIL (2D & 3D)

    def _log_phase_interference(self, trainer):
        if not mlflow:
            return
        model = trainer.model
        model.eval()

        if hasattr(trainer, "get_batch"):
            x, _ = trainer.get_batch("val")
        else:
            return

        x = x.to(trainer.device if hasattr(trainer, "device") else "cpu")

        sail_data = {}
        handles = []

        for layer_idx, block in enumerate(model.blocks):

            def make_hook(idx):

                def hook_fn(module, input, output):
                    sail_data[idx] = {
                        "z_in": input[0].detach(),
                        "z_out": output.detach(),
                    }

                return hook_fn

            handle = block.sail.register_forward_hook(make_hook(layer_idx))
            handles.append(handle)

        with torch.no_grad():
            _ = model(x)

        for handle in handles:
            handle.remove()

        num_layers = len(model.blocks)
        layer_metrics = {}

        for l_idx in range(num_layers):
            z_in = sail_data[l_idx]["z_in"]
            z_out = sail_data[l_idx]["z_out"]

            p_in = torch.angle(z_in).cpu().numpy().flatten()
            p_out = torch.angle(z_out).cpu().numpy().flatten()
            m_out = torch.abs(z_out).cpu().numpy().flatten()

            layer_metrics[l_idx] = (p_in, p_out, m_out)

        self.frames_data.append((trainer.step, layer_metrics))

        model.train()

    def on_train_start(self, trainer):
        if not mlflow:
            return
        mlflow.set_experiment(self.config.experiment)
        if self.config.system_metrics:
            mlflow.enable_system_metrics_logging()
        mlflow.start_run()

        if hasattr(trainer, "config"):
            mlflow.log_params(vars(trainer.config))
        if hasattr(trainer.model, "config"):
            mlflow.log_params(vars(trainer.model.config))

        if hasattr(trainer.model, "get_num_params"):
            num_params = trainer.model.get_num_params()
            mlflow.log_param("num_params", num_params)
            mlflow.log_param("num_params_M", f"{num_params / 1e6:.2f}M")

    def on_step(self, trainer, loss):
        if not mlflow:
            return

        if trainer.step % self.config.log_interval == 0:
            mlflow.log_metric("step loss", loss.item(), step=trainer.step)

        if (
            self.config.deep_log_interval > 0
            and trainer.step % self.config.deep_log_interval == 0
        ):
            try:
                self._log_phase_interference(trainer)
            except Exception as e:
                print(f"⚠️ Deep Eval Interference Plot skipped: {e}")

    def on_eval(self, trainer, losses):
        if not mlflow:
            return
        mlflow.log_metrics(
            {"train loss": losses["train"], "val loss": losses["val"]},
            step=trainer.step,
        )

    def on_train_end(self, trainer, losses):
        if not mlflow:
            return

        from matplotlib import animation
        from matplotlib import pyplot as plt

        mlflow.log_metrics(
            {"train loss": losses["train"], "val loss": losses["val"]},
            step=trainer.step,
        )

        num_layers = len(trainer.model.blocks)
        cols = 2 if num_layers > 1 else 1
        rows = math.ceil(num_layers / cols)

        #  2D MULTI-LAYER SAIL GIF
        if self.frames_data:
            print("🎬 Rendering 2D Multi-Layer SAIL Phase Interference GIF...")
            fig_2d, axes_2d = plt.subplots(
                rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
            )

            def update_2d(frame_idx):
                step, layer_metrics = self.frames_data[frame_idx]
                fig_2d.suptitle(
                    f"SAIL Interference Evolution - Step {step}",
                    fontsize=14,
                    fontweight="bold",
                )

                for l_idx in range(num_layers):
                    r, c = l_idx // cols, l_idx % cols
                    ax = axes_2d[r][c]
                    ax.clear()

                    p_in, p_out, m_out = layer_metrics[l_idx]
                    ax.hist2d(p_in, p_out, bins=100, weights=m_out, cmap="twilight")
                    ax.set_title(f"Layer {l_idx}", fontsize=11)
                    ax.set_xlabel("Input Phase $\\theta_{in}$")
                    ax.set_ylabel("Output Phase $\\theta_{out}$")

            ani_2d = animation.FuncAnimation(
                fig_2d, update_2d, frames=len(self.frames_data), interval=300
            )
            gif_path_2d = "examples/saved/sail_all_layers_evolution.gif"
            ani_2d.save(gif_path_2d, writer="pillow", fps=3)
            plt.close(fig_2d)
            mlflow.log_artifact(gif_path_2d, artifact_path="animations")

            # 2. 3D MULTI-LAYER SAIL TOPOGRAPHY GIF
            print("🎬 Rendering 3D Multi-Layer SAIL Phase-Amplitude Topography GIF...")
            fig_3d = plt.figure(figsize=(6 * cols, 5 * rows))

            def update_3d(frame_idx):
                fig_3d.clf()
                step, layer_metrics = self.frames_data[frame_idx]
                fig_3d.suptitle(
                    f"SAIL 3D Phase-Amplitude Evolution (Step {step})",
                    fontsize=14,
                    fontweight="bold",
                )

                for l_idx in range(num_layers):
                    p_in, p_out, m_out = layer_metrics[l_idx]
                    counts, xedges, yedges = np.histogram2d(
                        p_in, p_out, bins=80, weights=m_out
                    )
                    X, Y = np.meshgrid(
                        (xedges[:-1] + xedges[1:]) / 2,
                        (yedges[:-1] + yedges[1:]) / 2,
                    )

                    ax = fig_3d.add_subplot(rows, cols, l_idx + 1, projection="3d")
                    ax.plot_surface(
                        X,
                        Y,
                        counts.T,
                        cmap="twilight",
                        edgecolor="none",
                        alpha=0.9,
                        antialiased=True,
                    )
                    ax.set_title(f"Layer {l_idx}", fontsize=11)
                    ax.set_xlabel("$\\theta_{in}$")
                    ax.set_ylabel("$\\theta_{out}$")
                    ax.set_zlabel("Resonance")

                    azim_angle = (225 + frame_idx * 0.5) % 360
                    ax.view_init(elev=35, azim=azim_angle)

            ani_3d = animation.FuncAnimation(
                fig_3d, update_3d, frames=len(self.frames_data), interval=300
            )
            gif_path_3d = "examples/saved/sail_3d_topography_evolution.gif"
            ani_3d.save(gif_path_3d, writer="pillow", fps=3)
            plt.close(fig_3d)
            mlflow.log_artifact(gif_path_3d, artifact_path="animations")
