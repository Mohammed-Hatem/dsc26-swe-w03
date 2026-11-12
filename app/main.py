"""Rawaj Pay transfer API."""
import hashlib
import json
import math
import uuid
from typing import Annotated

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError
from sqlalchemy import select

from app.db import make_engine, make_session_factory
from app.models import Account, IdempotencyRecord, Transfer
from app.settings import Settings


class TransferRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_account_id: str
    to_account_id: str
    amount_piastres: StrictInt = Field(gt=0)
    memo: str | None = Field(default=None, max_length=140)


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self.capacity = settings.rate_limit_capacity
        self.refill = settings.rate_limit_refill_per_second
        self.clock = settings.clock
        self.buckets: dict[str, tuple[float, float]] = {}

    def take(self, customer_id: str) -> tuple[bool, int]:
        now = self.clock()
        tokens, previous = self.buckets.get(customer_id, (float(self.capacity), now))
        tokens = min(float(self.capacity), tokens + (now - previous) * self.refill)
        if tokens < 1:
            wait = math.ceil((1 - tokens) / self.refill) if self.refill > 0 else 1
            self.buckets[customer_id] = (tokens, now)
            return False, max(1, wait)
        self.buckets[customer_id] = (tokens - 1, now)
        return True, 0


def _problem(
    status: int,
    title: str,
    detail: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = {"type": f"https://httpstatuses.com/{status}", "title": title, "status": status}
    if detail is not None:
        body["detail"] = detail
    return JSONResponse(body, status_code=status, media_type="application/problem+json", headers=headers)


def _transfer_body(transfer: Transfer) -> dict[str, object]:
    return {
        "id": transfer.id,
        "from_account_id": transfer.from_account_id,
        "to_account_id": transfer.to_account_id,
        "amount_piastres": transfer.amount_piastres,
        "memo": transfer.memo,
        "status": transfer.status,
    }


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI()
    engine = make_engine(settings.database_url)
    app.state.engine = engine
    session_factory = make_session_factory(engine)
    limiter = RateLimiter(settings)

    def authenticate(authorization: str | None) -> str | JSONResponse:
        if authorization is None:
            return _problem(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        scheme, separator, token = authorization.partition(" ")
        if scheme != "Bearer" or not separator or not token or token not in settings.api_tokens:
            return _problem(401, "Unauthorized", headers={"WWW-Authenticate": "Bearer"})
        return settings.api_tokens[token]

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(_: Request, exc: ValidationError) -> JSONResponse:
        return _problem(422, "Unprocessable Entity", str(exc))

    @app.post("/v1/transfers")
    async def create_transfer(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> JSONResponse:
        customer = authenticate(authorization)
        if isinstance(customer, JSONResponse):
            return customer
        allowed, retry_after = limiter.take(customer)
        if not allowed:
            return _problem(429, "Too Many Requests", headers={"Retry-After": str(retry_after)})

        try:
            payload = TransferRequest.model_validate(await request.json())
        except (ValidationError, ValueError, TypeError) as exc:
            return _problem(422, "Unprocessable Entity", str(exc))
        if payload.amount_piastres > settings.max_transfer_piastres:
            return _problem(422, "Unprocessable Entity", "amount exceeds the configured maximum")
        if idempotency_key is None:
            return _problem(400, "Bad Request", "Idempotency-Key is required")
        body = payload.model_dump(mode="json")
        request_hash = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

        with session_factory.begin() as session:
            existing = session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.customer_id == customer,
                    IdempotencyRecord.key == idempotency_key,
                )
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    return _problem(422, "Unprocessable Entity", "idempotency key was reused with a different body")
                return JSONResponse(
                    json.loads(existing.response_body),
                    status_code=existing.response_status,
                    headers={"Idempotent-Replayed": "true"},
                )

            source = session.get(Account, payload.from_account_id)
            if source is None or source.owner_id != customer:
                return _problem(404, "Not Found", "source account was not found")
            destination = session.get(Account, payload.to_account_id)
            if destination is None or payload.from_account_id == payload.to_account_id:
                return _problem(422, "Unprocessable Entity", "destination account is invalid")
            if source.balance_piastres < payload.amount_piastres:
                response = _problem(409, "Conflict", "insufficient funds")
                session.add(
                    IdempotencyRecord(
                        customer_id=customer,
                        key=idempotency_key,
                        request_hash=request_hash,
                        response_status=409,
                        response_body=response.body.decode(),
                    )
                )
                return response

            source.balance_piastres -= payload.amount_piastres
            destination.balance_piastres += payload.amount_piastres
            transfer = Transfer(
                id=f"trf_{uuid.uuid4().hex}",
                from_account_id=payload.from_account_id,
                to_account_id=payload.to_account_id,
                amount_piastres=payload.amount_piastres,
                memo=payload.memo,
                status="completed",
            )
            session.add(transfer)
            result = _transfer_body(transfer)
            session.add(
                IdempotencyRecord(
                    customer_id=customer,
                    key=idempotency_key,
                    request_hash=request_hash,
                    response_status=201,
                    response_body=json.dumps(result),
                )
            )
            return JSONResponse(result, status_code=201)

    @app.get("/v1/accounts/{account_id}")
    async def get_account(
        account_id: str, authorization: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        customer = authenticate(authorization)
        if isinstance(customer, JSONResponse):
            return customer
        with session_factory() as session:
            account = session.get(Account, account_id)
            if account is None or account.owner_id != customer:
                return _problem(404, "Not Found", "account was not found")
            return JSONResponse(
                {"id": account.id, "balance_piastres": account.balance_piastres, "currency": account.currency}
            )

    @app.get("/v1/transfers/{transfer_id}")
    async def get_transfer(
        transfer_id: str, authorization: Annotated[str | None, Header()] = None
    ) -> JSONResponse:
        customer = authenticate(authorization)
        if isinstance(customer, JSONResponse):
            return customer
        with session_factory() as session:
            transfer = session.get(Transfer, transfer_id)
            if transfer is None:
                return _problem(404, "Not Found", "transfer was not found")
            source = session.get(Account, transfer.from_account_id)
            destination = session.get(Account, transfer.to_account_id)
            if customer not in {source.owner_id if source else None, destination.owner_id if destination else None}:
                return _problem(404, "Not Found", "transfer was not found")
            return JSONResponse(_transfer_body(transfer))

    return app
