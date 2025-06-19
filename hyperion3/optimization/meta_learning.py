class MAML:
    """Very small MAML-like routine."""

    def __init__(self, config):
        self.config = config
        self.inner_lr = (
            getattr(config, "optimization", config).maml_inner_lr
            if hasattr(config, "optimization")
            else 0.01
        )

    async def adapt(self, task_data):
        losses = []
        for x, y in task_data:
            loss = (x - y) ** 2
            losses.append(float(loss))

        mean_loss = sum(losses) / len(losses) if losses else 0.0
        return {"loss": mean_loss}


__all__ = ["MAML"]
