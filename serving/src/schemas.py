"""API 요청/응답 스키마 — spec §2. 요청 불변식은 여기서 강제한다."""

import math
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # strict=True: bool·"1"·1.0의 묵시적 int 변환 금지
    units_held: int = Field(ge=0, strict=True)
    shares_held: float = Field(ge=0)
    cost_basis: Optional[float] = Field(default=None, ge=0)
    bars_since_entry: int = Field(ge=0, strict=True)
    available_cash: float = Field(ge=0)

    @model_validator(mode="after")
    def _invariants(self) -> "PortfolioState":
        for name in ("shares_held", "available_cash", "cost_basis"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (self.units_held == 0) != (self.shares_held == 0):
            raise ValueError("units_held == 0 must coincide with shares_held == 0")
        if self.units_held == 0 and self.bars_since_entry != 0:
            raise ValueError("bars_since_entry must be 0 while flat")
        if self.units_held == 0 and self.cost_basis not in (None, 0.0):
            raise ValueError("cost_basis must be 0 or omitted while flat")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1)
    portfolio: PortfolioState
    as_of: Optional[datetime] = None


class PredictResponse(BaseModel):
    action: int
    label: str
    action_mask: List[bool]
    bar_ts: datetime
    artifact_id: str
    feature_schema_version: int
    observation: List[float]
    bar_close: float
