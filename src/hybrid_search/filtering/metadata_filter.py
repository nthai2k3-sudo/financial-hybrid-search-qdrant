"""Translates MetadataFilter models to native Qdrant filter conditions."""

from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timezone
from qdrant_client import models
from hybrid_search.models.filter import MetadataFilter


class QdrantFilterBuilder:
    """Constructs Qdrant models.Filter instances from MetadataFilter constraints."""

    @classmethod
    def _parse_date(cls, date_str: str) -> datetime:
        """Parses YYYY-MM-DD string into UTC datetime."""
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)

    @classmethod
    def build_filter(cls, filter_spec: Optional[MetadataFilter]) -> Optional[models.Filter]:
        """Builds a composite Qdrant Filter object (AND conditions)."""
        if filter_spec is None or filter_spec.is_empty():
            return None

        must_conditions: List[models.Condition] = []

        # Tickers
        if filter_spec.tickers:
            if len(filter_spec.tickers) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchValue(value=filter_spec.tickers[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="ticker",
                        match=models.MatchAny(any=filter_spec.tickers),
                    )
                )

        # Document types
        if filter_spec.document_types:
            if len(filter_spec.document_types) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchValue(value=filter_spec.document_types[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchAny(any=filter_spec.document_types),
                    )
                )

        # Section
        if filter_spec.sections:
            if len(filter_spec.sections) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="section",
                        match=models.MatchValue(value=filter_spec.sections[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="section",
                        match=models.MatchAny(any=filter_spec.sections),
                    )
                )

        # Sector
        if filter_spec.sectors:
            if len(filter_spec.sectors) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="sector",
                        match=models.MatchValue(value=filter_spec.sectors[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="sector",
                        match=models.MatchAny(any=filter_spec.sectors),
                    )
                )

        # Geography
        if filter_spec.geographies:
            if len(filter_spec.geographies) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="geography",
                        match=models.MatchValue(value=filter_spec.geographies[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="geography",
                        match=models.MatchAny(any=filter_spec.geographies),
                    )
                )

        # Fiscal periods
        if filter_spec.fiscal_periods:
            if len(filter_spec.fiscal_periods) == 1:
                must_conditions.append(
                    models.FieldCondition(
                        key="fiscal_period",
                        match=models.MatchValue(value=filter_spec.fiscal_periods[0]),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key="fiscal_period",
                        match=models.MatchAny(any=filter_spec.fiscal_periods),
                    )
                )

        # Date range filtering using DatetimeRange
        if filter_spec.filing_date_start or filter_spec.filing_date_end:
            gte_dt = cls._parse_date(filter_spec.filing_date_start) if filter_spec.filing_date_start else None
            lte_dt = cls._parse_date(filter_spec.filing_date_end) if filter_spec.filing_date_end else None

            must_conditions.append(
                models.FieldCondition(
                    key="filing_date",
                    range=models.DatetimeRange(
                        gte=gte_dt,
                        lte=lte_dt,
                    ),
                )
            )

        if not must_conditions:
            return None

        return models.Filter(must=must_conditions)
