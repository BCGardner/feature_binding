from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload, sessionmaker, Session

from hsnn.core.logger import logging
from hsnn.utils.handler import TrialView, get_hfb_path
from ..base import PNG
from .models import Base, PNGModel, LagModel, RunModel
from . import _utils

__all__ = ["PNGDatabase"]

ModelType = TypeVar('ModelType', bound=Base)

logger = logging.getLogger()


class PNGDatabase:
    _instances: dict[str, PNGDatabase] = {}
    _initialised: bool = False
    _db_path: str

    def __new__(cls, db_path: str | Path):
        db_path = str(Path(db_path).resolve())

        if db_path not in cls._instances:
            cls._instances[db_path] = super(PNGDatabase, cls).__new__(cls)
        return cls._instances[db_path]

    def __init__(self, db_path: str | Path, **kwargs) -> None:
        self._db_path = str(Path(db_path).resolve())
        self.connect(**kwargs)

    @staticmethod
    def from_trial(trial: TrialView, chkpt_idx: Optional[int] = None, *,
                   subdir: Optional[str] = None, sgnf: bool = False,
                   engine_kwargs: Optional[dict] = None, **kwargs) -> PNGDatabase:
        db_path = get_hfb_path(trial, chkpt_idx, subdir=subdir, sgnf=sgnf, **kwargs)
        engine_kwargs = engine_kwargs or {}
        return PNGDatabase(db_path, **engine_kwargs)

    @property
    def path(self) -> str:
        return self._db_path

    @property
    def exists(self) -> bool:
        return os.path.exists(self._db_path)

    def connect(self, **kwargs) -> None:
        if not self._initialised:
            self._engine = create_engine(f'sqlite:///{self._db_path}', **kwargs)
            self._initialised = True
            logger.info(f"Connected to {self._engine}")

    def close(self) -> None:
        self._engine.dispose()
        self._initialised = False

    def create(self) -> None:
        Base.metadata.create_all(self._engine)

    def get_session(self, **kwargs) -> Session:
        Session = sessionmaker(bind=self._engine, **kwargs)
        return Session()

    def insert_pngs(self, pngs: Sequence[PNG]):
        with self.get_session() as session:
            try:
                instances = []
                for polygrp in pngs:
                    existing_model = session.query(PNGModel).filter(
                        PNGModel.id == hash(polygrp)).first()
                    if existing_model is None:
                        instances.append(_utils.create_png_model(polygrp))
                    else:
                        logging.warning(f"Skipped for {existing_model}")
                session.add_all(instances)
                session.commit()
            except Exception as exc:
                session.rollback()
                raise exc

    def insert_runs(self, nrn_ids: Iterable[int], layer: int, index: int):
        with self.get_session() as session:
            try:
                instances = []
                for nrn_id in set(nrn_ids):
                    existing_model = session.query(RunModel).filter(
                        RunModel.neuron == int(nrn_id),
                        RunModel.layer == int(layer),
                        RunModel.index == int(index)).first()
                    if existing_model is None:
                        instances.append(_utils.create_run_model(nrn_id, layer, index))
                    else:
                        logging.warning(f"Skipped for {existing_model}")
                session.add_all(instances)
                session.commit()
            except IntegrityError as exc:
                session.rollback()
                raise exc

    def get_records(self, model_type: Type[ModelType],
                    filters: Optional[Iterable[Any]] = None) -> list[ModelType]:
        with self.get_session() as session:
            query = session.query(model_type)
            if model_type == PNGModel:
                query = query.options(joinedload(PNGModel.lags), joinedload(PNGModel.onsets))
            if filters:
                query = query.filter(*filters)
            return query.all()

    def get_pngs(self, layer: int, nrn_ids: Iterable[int],
                 index: Optional[int] = None) -> Sequence[PNG]:
        assert isinstance(layer, int)
        assert isinstance(next(iter(nrn_ids)), int)
        with self.get_session() as session:
            stmt = (
                select(PNGModel)
                .join(PNGModel.lags)
                .where(LagModel.layer == layer, LagModel.neuron.in_(nrn_ids))
            )
            if index is not None:
                assert isinstance(index, int)
                stmt = stmt.where(LagModel.index == index)
            entries = session.scalars(stmt).all()
            polygrps = _utils.recreate_all(entries)
        return polygrps

    def get_all_pngs(self) -> Sequence[PNG]:
        with self.get_session() as session:
            png_models = session.query(PNGModel).all()
            polygrps = _utils.recreate_all(png_models)
        return polygrps

    def get_run_nrns(self, layer: int, index: int) -> Sequence[int]:
        assert isinstance(layer, int)
        assert isinstance(index, int)
        with self.get_session() as session:
            stmt = (
                select(RunModel)
                .where(RunModel.layer == layer,
                       RunModel.index == index)
            )
            entries = session.scalars(stmt).all()
            nrn_ids = [entry.neuron for entry in entries]
        return sorted(nrn_ids)

    def remove(self, png_models: Sequence[PNGModel]):
        session = self.get_session()
        try:
            for png_model in png_models:
                session.delete(png_model)
            session.commit()
        except Exception as exc:
            session.rollback()
            raise exc
        finally:
            session.close()
