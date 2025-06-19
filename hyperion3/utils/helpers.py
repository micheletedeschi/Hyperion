"""Funciones auxiliares varias para la librería."""

from typing import Iterable, List


def chunkify(data: Iterable, size: int) -> List[List]:
    """Divide un iterable en listas de tamaño fijo.

    Args:
        data: Iterable de entrada.
        size: Número máximo de elementos por cada chunk.

    Yields:
        Lista de elementos de longitud ``size`` (el último puede ser menor).
    """

    chunk = []
    for item in data:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


__all__ = ["chunkify"]
