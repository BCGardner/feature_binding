from typing import Optional

import numpy as np
from sqlalchemy import ForeignKey
from sqlalchemy.orm import mapped_column, relationship, DeclarativeBase, Mapped

__all__ = ["Base", "PNGModel", "LagModel", "OnsetModel", "RunModel"]


class Base(DeclarativeBase):
    ...


class PNGModel(Base):
    __tablename__ = 'png'

    id: Mapped[int] = mapped_column(primary_key=True)
    size: Mapped[int]
    span: Mapped[float]

    lags: Mapped[list["LagModel"]] = relationship(
        back_populates="png", cascade="all, delete-orphan"
    )
    onsets: Mapped[list["OnsetModel"]] = relationship(
        back_populates="png", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        layers = np.array([lag.layer for lag in self.lags])
        neurons = np.array([lag.neuron for lag in self.lags])
        lags = np.array([lag.time for lag in self.lags])
        args = np.lexsort((neurons, layers, lags))
        occurrences = len(self.onsets)
        return (f"PNG(layers={layers[args]}, neurons={neurons[args]}, "
                f"lags={lags[args]}, occs={occurrences})")


class LagModel(Base):
    __tablename__ = 'lag'

    png_id: Mapped[int] = mapped_column(ForeignKey('png.id'), primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)
    layer: Mapped[int]
    neuron: Mapped[int]
    time: Mapped[float]

    png: Mapped["PNGModel"] = relationship(back_populates="lags")

    def __repr__(self) -> str:
        return (f"Lag(png_id={self.png_id}, index={self.index}, layer={self.layer}, "
                f"neuron={self.neuron}, time={self.time})")


class OnsetModel(Base):
    __tablename__ = 'onset'

    png_id: Mapped[int] = mapped_column(ForeignKey('png.id'), primary_key=True)
    time: Mapped[float] = mapped_column(primary_key=True)
    img: Mapped[Optional[int]]
    rep: Mapped[Optional[int]]
    time_rel: Mapped[Optional[float]]

    png: Mapped["PNGModel"] = relationship(back_populates="onsets")

    def __repr__(self) -> str:
        return f"Onset(png_id={self.png_id}, time={self.time})"


class RunModel(Base):
    __tablename__ = 'run'

    # config_id: Mapped[int] = mapped_column(ForeignKey('config.id'))
    layer: Mapped[int] = mapped_column(primary_key=True)
    neuron: Mapped[int] = mapped_column(primary_key=True)
    index: Mapped[int] = mapped_column(primary_key=True)

    # config: Mapped["ConfigModel"] = relationship(back_populates="runs")

    def __repr__(self) -> str:
        return (f"Run(layer={self.layer}, neuron={self.neuron}, "
                f"index={self.index})")


# class ConfigModel(Base):
#     __tablename__ = 'config'

#     id: Mapped[int] = mapped_column(primary_key=True)
#     duration: Mapped[float]
#     offset: Mapped[float]
#     separation: Mapped[float]
#     num_reps: Mapped[int]

#     runs: Mapped[list["RunModel"]] = relationship(back_populates="config")

#     def __repr__(self) -> str:
#         return (f"Config(duration={self.duration}, offset={self.offset}, "
#                 f"separation={self.separation}, num_reps={self.num_reps})")
