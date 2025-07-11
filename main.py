from typing import final

from numpy import float32
from torch import Tensor
from faiss import IndexFlatL2 # pyright: ignore[reportMissingTypeStubs]
from lmdbm import Lmdb
from msgpack import packb, unpackb # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

@final
class Database:
  def __init__(self):
    self.database: Lmdb[bytes, bytes] = Lmdb.open("LESCdb.lmdb", "c") # pyright: ignore[reportUnknownMemberType]
    self.staging: Lmdb[bytes, bytes] = Lmdb.open("LESCdb_staging.lmdb", "c") # pyright: ignore[reportUnknownMemberType]
    self.index = IndexFlatL2(128)
    self.evaluations: list[list[float]] = []

  def _insert_db(self, db: Lmdb[bytes, bytes], key: list[float], value: list[float]):
    serialized: bytes | None = packb(key) # pyright: ignore[reportUnknownVariableType]
    if serialized is None:
      raise ValueError("Key cannot be serialized")
    serialized_value = packb(value) # pyright: ignore[reportUnknownVariableType]
    if serialized_value is None:
      raise ValueError("Value cannot be serialized")
    db[serialized] = serialized_value

  def _select_db(self, db: Lmdb[bytes, bytes], key: list[float]) -> list[float]:
    serialized: bytes | None = packb(key) # pyright: ignore[reportUnknownVariableType]
    if serialized is None:
      raise ValueError("Key cannot be serialized")
    return unpackb(db[serialized]) # pyright: ignore[reportUnknownVariableType]

  def insert_staging(self, key: list[float], value: list[float]):
    self._insert_db(self.staging, key, value)

  def _train(self, original: Tensor, data: Tensor) -> None:
    pass

  def _evaluate(self, data: Tensor) -> Tensor:
    return data

  def build(self):
    original = Tensor()
    data = Tensor()
    self._train(original, data)
    i = 0
    eval_result = self._evaluate(data)
    le = len(eval_result)
    while i < le:
      current_data: list[float] = data[i].tolist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
      current_eval: list[float] = eval_result[i].tolist() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
      self._insert_db(self.database, current_eval, current_data)
      self.index.add(eval_result[i].numpy()) # pyright: ignore[reportCallIssue, reportUnknownMemberType]
      self.evaluations.append(current_eval)
      i += 1

  def search(self, value: list[float], limit: int = 10) -> list[tuple[float, list[float]]]:
    results: list[tuple[float, list[float]]] = []
    evaluation = self._evaluate(Tensor([value]))
    distances, indices = self.index.search(evaluation.numpy(), limit) # pyright: ignore[reportCallIssue, reportUnknownMemberType, reportUnknownVariableType]
    for distance, indice in zip(distances, indices): # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
      distance: float32
      indice: int
      results.append((float(distance), self._select_db(self.database, self.evaluations[indice]))) # pyright: ignore[reportUnknownArgumentType]
    return results
