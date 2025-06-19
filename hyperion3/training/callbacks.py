class TrainingCallback:
    """Simple callback interface."""

    def on_epoch_end(self, epoch: int, logs: dict):
        msg_parts = [f"Epoch {epoch}"]
        for key, value in logs.items():
            if isinstance(value, float):
                msg_parts.append(f"{key}={value:.4f}")
            else:
                msg_parts.append(f"{key}={value}")
        print(" - ".join(msg_parts))


__all__ = ["TrainingCallback"]
