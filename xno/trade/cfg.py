import logging
import uuid
from typing import Any, Iterable

from cachetools.func import ttl_cache
from pydantic import BaseModel
from sqlalchemy import text

from xno import settings
from xno.connectors.semaphore import DistributedSemaphore
from xno.connectors.sql import SqlSession
from xno.trade.tp import AllowedTradeMode, AllowedEngine
from contextlib import ExitStack


class AdvancedConfig(BaseModel):
    expression: str = ""
    close_on_end_day: bool = False
    use_trailing_stop: bool = False
    trailing_stop_pct: float = 0

class StrategyConfig(BaseModel):
    strategy_id: str
    symbol: str
    symbol_type: str
    timeframe: str
    init_cash: float
    run_from: Any
    run_to: Any
    mode: AllowedTradeMode
    advanced_config: AdvancedConfig
    engine: str

class StrategyConfigLoader:
    @classmethod
    def get_live_strategy_configs(cls, symbol_type) -> Iterable[StrategyConfig]:
        query = f"""
            SELECT
                id,
                symbol,
                timeframe,
                live as result,
                advanced_config as advanced_config,
                engine as engine
            FROM public.strategy_overview
            WHERE engine = :engine AND symbol_type = :symbol_type
            ORDER BY symbol, id
        """
        with ExitStack() as stack:
            # Lock to ensure thread-safe read
            stack.enter_context(DistributedSemaphore())
            session = stack.enter_context(SqlSession(settings.execution_db_name))
            result_iter = session.execute(
                text(query),
                {"engine": AllowedEngine.TABot, "symbol_type": symbol_type},
            )

        # Stream rows as they come (no fetchall)
        for row in result_iter:
            run_from = row.result.get('from')
            run_to = row.result.get('to')
            init_cash = row.result.get('cash')

            if run_from is None or run_to is None or init_cash is None:
                logging.warning(
                    f"Skipping strategy {row.id} due to missing run_from, run_to, or init_cash in live config."
                )
                continue

            yield StrategyConfig(
                strategy_id=str(row.id),
                symbol=row.symbol,
                symbol_type=symbol_type,
                timeframe=row.timeframe,
                init_cash=init_cash,
                run_from=run_from,
                run_to=run_to,
                mode=AllowedTradeMode.LiveTrade,
                advanced_config=AdvancedConfig(**row.advanced_config),
                engine=row.engine,
            )

    @classmethod
    @ttl_cache(ttl=3600 * 8, maxsize=1000000)  # Cache for 8 hours
    def get_config(cls, strategy_id: str, mode: AllowedTradeMode, run_id="") -> StrategyConfig | None:
        logging.info(f"Getting config for strategy_id={strategy_id}, mode={mode}, run_id={run_id}")
        if mode == AllowedTradeMode.BackTrade:
            column = "backtest"
        elif mode == AllowedTradeMode.PaperTrade:
            column = "paper"
        else:
            column = "live"

        query = f"""
            SELECT
                id,
                symbol,
                symbol_type as symbol_type,
                timeframe,
                {column} as result,
                advanced_config as advanced_config,
                engine as engine
            FROM public.strategy_overview
            WHERE id = :strategy_id AND engine = :engine
            LIMIT 1
        """
        # Lock to ensure thread-safe read
        with ExitStack() as stack:
            stack.enter_context(DistributedSemaphore())
            session = stack.enter_context(SqlSession(settings.execution_db_name))
            result = session.execute(
                text(query),
                {'strategy_id': strategy_id, "engine": AllowedEngine.TABot},
            )

        row = result.fetchone()
        if row is None:
            return None

        run_from = row.result['from']
        run_to = row.result['to']
        init_cash = row.result['cash']
        # Return the config with the appropriate run_from and run_to
        return StrategyConfig(
            strategy_id=row.id.__str__(),
            symbol=row.symbol,
            symbol_type=row.symbol_type,
            timeframe=row.timeframe,
            init_cash=init_cash,
            run_from=run_from,
            run_to=run_to,
            mode=mode,
            advanced_config=AdvancedConfig(**row.advanced_config),
            engine=row.engine,
        )


if __name__ == "__main__":
    # Initial load
    # Example usage
    config = StrategyConfigLoader.get_config("fad40f3b-52a7-44d1-99cb-8d4b5aa257c5", AllowedTradeMode.BackTrade, "")
    print(config)
    print(config.model_dump_json())  # to string

    configs = StrategyConfigLoader.get_live_strategy_configs("S")
    print("Live strategy config len:", len(list(configs)))

    config = StrategyConfigLoader.get_config(uuid.uuid4().__str__(), AllowedTradeMode.BackTrade, "")
    print(config)
    print(config.model_dump_json())  # to string
