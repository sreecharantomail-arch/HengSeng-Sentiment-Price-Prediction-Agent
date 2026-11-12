"""
Core Apify integration layer.

Provides :class:`ApifyClient` wrapper with actor-run execution, result
retrieval, and automatic token management via :class:`APIConfig`.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from apify_client import ApifyClient as _ApifyClient

from src.config import APIConfig

logger = logging.getLogger(__name__)

# Default Apify actor timeout (seconds)
_DEFAULT_TIMEOUT_SECS = 300


class ApifyRunError(Exception):
    """Raised when an Apify actor run fails or times out."""


class ApifyRunner:
    """Thin async wrapper around the Apify REST client.

    Usage::

        runner = ApifyRunner()
        items = await runner.run_actor(
            actor_id="apify/web-scraper",
            run_input={"startUrls": [...]},
        )
    """

    def __init__(self, *, api_config: APIConfig | None = None) -> None:
        cfg = api_config or APIConfig()
        if not cfg.apify_token:
            raise ValueError(
                "APIFY_API_TOKEN is required. Set it in your .env file."
            )
        self._client = _ApifyClient(token=cfg.apify_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_actor(
        self,
        actor_id: str,
        run_input: dict[str, Any],
        *,
        timeout_secs: int = _DEFAULT_TIMEOUT_SECS,
        memory_mbytes: int | None = None,
        build: str | None = None,
    ) -> list[dict[str, Any]]:
        """Start an Apify actor run, wait for it, and return dataset items.

        Parameters
        ----------
        actor_id:
            Qualified actor name, e.g. ``"apify/web-scraper"``.
        run_input:
            JSON-serialisable input dict for the actor.
        timeout_secs:
            Maximum wall-clock seconds to wait for the run.
        memory_mbytes:
            Optional memory allocation override (128–32768).
        build:
            Optional actor build tag (default: ``"latest"``).

        Returns
        -------
        list[dict]
            Raw dataset items produced by the actor run.

        Raises
        ------
        ApifyRunError
            If the run fails, is aborted, or times out.
        """
        logger.info("Starting Apify actor %s", actor_id)
        logger.debug("Run input: %s", run_input)

        call_kwargs: dict[str, Any] = {
            "run_input": run_input,
            "timeout_secs": timeout_secs,
        }
        if memory_mbytes is not None:
            call_kwargs["memory_mbytes"] = memory_mbytes
        if build is not None:
            call_kwargs["build"] = build

        try:
            # apify_client is synchronous – offload to a thread
            run_info = await asyncio.to_thread(
                self._client.actor(actor_id).call, **call_kwargs
            )
        except Exception as exc:
            raise ApifyRunError(
                f"Actor {actor_id} call failed: {exc}"
            ) from exc

        status = run_info.get("status", "UNKNOWN")
        if status != "SUCCEEDED":
            raise ApifyRunError(
                f"Actor {actor_id} finished with status {status}"
            )

        dataset_id = run_info.get("defaultDatasetId", "")
        if not dataset_id:
            logger.warning("Actor %s returned no dataset ID", actor_id)
            return []

        try:
            items: list[dict[str, Any]] = await asyncio.to_thread(
                lambda: list(
                    self._client.dataset(dataset_id).iterate_items()
                )
            )
        except Exception as exc:
            raise ApifyRunError(
                f"Failed to fetch dataset {dataset_id}: {exc}"
            ) from exc

        logger.info(
            "Actor %s returned %d items (dataset %s)",
            actor_id,
            len(items),
            dataset_id,
        )
        return items
