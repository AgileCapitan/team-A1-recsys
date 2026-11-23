from __future__ import annotations

from typing import Any

import requests


class APIError(Exception):
    """Raised when communicating with the recommender FastAPI service fails."""

    def __init__(self, detail: str, status_code: int | None = None) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _extract_error_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"Backend returned HTTP {response.status_code}."

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list) and detail:
        messages = []
        for err in detail:
            if isinstance(err, dict):
                loc = ".".join(str(part) for part in err.get("loc", []))
                msg = err.get("msg", "Invalid input")
                messages.append(f"{loc}: {msg}" if loc else str(msg))
            else:
                messages.append(str(err))
        return "; ".join(messages)
    return f"Backend returned HTTP {response.status_code}."


class RecommenderAPIClient:
    """HTTP client encapsulating communication with the FastAPI recommender backend."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | list[tuple[str, Any]] | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(
                method,
                url,
                json=json,
                params=params,
                timeout=self.timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise APIError(f"Request to {url} timed out after {self.timeout}s.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise APIError(f"Could not connect to backend at {self.base_url}.") from exc
        except requests.exceptions.RequestException as exc:
            raise APIError(f"HTTP request failed: {exc}") from exc

        if response.status_code >= 400:
            raise APIError(
                _extract_error_detail(response),
                status_code=response.status_code,
            )
        return response

    def get_health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 5,
        genres: list[str] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"user_id": int(user_id), "top_k": int(top_k)}
        if genres:
            payload["genres"] = list(genres)
        return self._request("POST", "/recommend", json=payload).json()

    def get_genres(self) -> dict[str, Any]:
        return self._request("GET", "/catalog/genres").json()

    def get_rating_options(
        self,
        genres: list[str],
        per_genre: int = 25,
    ) -> dict[str, Any]:
        params: list[tuple[str, Any]] = [("genres", genre) for genre in genres]
        params.append(("per_genre", int(per_genre)))
        return self._request("GET", "/catalog/options", params=params).json()

    def recommend_new_user(
        self,
        genres: list[str],
        ratings: dict[int, int],
        top_k: int = 5,
    ) -> dict[str, Any]:
        payload = {
            "genres": list(genres),
            "ratings": {str(int(movie_id)): int(rating) for movie_id, rating in ratings.items()},
            "top_k": int(top_k),
        }
        return self._request("POST", "/recommend/new-user", json=payload).json()

    def get_poster_bytes(self, movie_id: int) -> bytes | None:
        url = f"{self.base_url}/posters/{int(movie_id)}"
        try:
            response = self.session.request("GET", url, timeout=self.timeout)
        except requests.exceptions.RequestException:
            return None
        if response.status_code == 200:
            return response.content
        return None
