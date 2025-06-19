"""Trading Environment Implementation for Hyperion V2"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class TradingEnvironmentSAC:
    def _get_state(self) -> np.ndarray:
        """
        Obtiene el estado actual del entorno.

        Returns:
            np.ndarray con el estado actual
        """
        try:
            # Obtener datos históricos
            start_idx = max(0, self.current_step - self.lookback_window)
            lookback_data = self.market_data.iloc[start_idx : self.current_step]

            # Si no hay suficientes datos históricos, usar padding
            if len(lookback_data) < self.lookback_window:
                padding = pd.DataFrame(
                    np.zeros(
                        (
                            self.lookback_window - len(lookback_data),
                            len(self.feature_columns),
                        )
                    ),
                    columns=self.feature_columns,
                )
                lookback_data = pd.concat([padding, lookback_data])

            # Convertir a array y aplanar
            state = lookback_data[self.feature_columns].values.flatten()

            # Añadir estado del portfolio
            current_price = self.market_data.iloc[self.current_step]["close"]
            portfolio_state = np.array(
                [
                    self.balance / self.initial_balance,  # Balance normalizado
                    self.current_position,  # Posición actual
                    current_price,  # Precio actual
                    self.current_position * current_price,  # Valor de la posición
                ]
            )

            return np.concatenate([state, portfolio_state])

        except Exception as e:
            logger.error(f"Error obteniendo estado: {str(e)}")
            raise
