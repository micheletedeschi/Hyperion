class TrainingCallback:
    """Simple callback interface."""

    def on_epoch_end(self, epoch: int, logs: dict):
        msg_parts = [f"Epoch {epoch}"]
        for key, value in logs.items():
            if isinstance(value, (int, float)) and value is not None:
                import numpy as np
                if not (np.isnan(value) or np.isinf(value)):
                    msg_parts.append(f"{key}={value:.4f}")
                else:
                    msg_parts.append(f"{key}=N/A")
            else:
                msg_parts.append(f"{key}={value}")
        print(" - ".join(msg_parts))


__all__ = ["TrainingCallback"]
