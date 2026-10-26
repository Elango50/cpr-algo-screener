import asyncio
from pathlib import Path
import threading
from typing import Any

from neo_api_client import NeoAPI
from neo_api_client.websocket.feed import WsToken, SFeedScrip
from dotenv import load_dotenv
import os

load_dotenv(Path(__file__).with_name(".env"))

live_market_data = {}

NIFTY_100_SYMBOLS = (
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "DMART", "AXISBANK", "BSE",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA",
    "BEL", "BPCL", "BHARTIARTL", "BOSCHLTD", "BRITANNIA", "CGPOWER", "CANBK",
    "CHOLAFIN", "CIPLA", "COALINDIA", "CUMMINSIND", "DLF", "DIVISLAB",
    "DRREDDY", "EICHERMOT", "ETERNAL", "GAIL", "GODREJCP", "GRASIM", "HCLTECH",
    "HDFCAMC", "HDFCBANK", "HDFCLIFE", "HINDALCO", "HAL", "HINDUNILVR",
    "HINDZINC", "POWERINDIA", "HYUNDAI", "ICICIBANK", "ITC", "IOC", "IRFC",
    "INFY", "INDIGO", "JSWSTEEL", "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTM",
    "LT", "M&M", "MARUTI", "MAXHEALTH", "MAZDOCK", "MUTHOOTFIN", "NTPC",
    "NESTLEIND", "ONGC", "PIDILITIND", "POLYCAB", "PFC", "POWERGRID", "PNB",
    "RELIANCE", "SBILIFE", "MOTHERSON", "SHRIRAMFIN", "ENRIN", "SIEMENS",
    "SOLARINDS", "SBIN", "SUNPHARMA", "TVSMOTOR", "TATACAP", "TCS",
    "TATACONSUM", "TMCV", "TMPV", "TATAPOWER", "TATASTEEL", "TECHM", "TITAN",
    "TORNTPHARM", "TRENT", "ULTRACEMCO", "UNIONBANK", "VBL", "VAML", "VEDL",
    "IDEA", "WIPRO", "ZYDUSLIFE",
)
NIFTY_TICKERS = tuple(f"{symbol}.NS" for symbol in NIFTY_100_SYMBOLS) + ("^NSEI",)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable {name}")
    return value


def _require_auth_data(response: dict[str, Any], step: str) -> dict[str, Any]:
    data = response.get("data")
    if isinstance(data, dict) and data.get("token") and data.get("sid"):
        return data

    errors = response.get("error")
    if isinstance(errors, list):
        messages = [
            error.get("message")
            for error in errors
            if isinstance(error, dict) and error.get("message")
        ]
        if messages:
            raise RuntimeError(f"{step} failed: {'; '.join(messages)}")

    raise RuntimeError(f"{step} failed: response did not include session credentials")


def _get_nifty_100_tokens(client: NeoAPI) -> list[WsToken]:
    symbols = set(NIFTY_100_SYMBOLS)

    scrips = client.search_scrip(exchange_segment="nse_cm")
    if not isinstance(scrips, list):
        raise RuntimeError(f"Failed to load NSE scrip master: {scrips}")

    tokens_by_symbol = {}
    for scrip in scrips:
        if not isinstance(scrip, dict):
            continue

        symbol = str(scrip.get("pSymbolName", "")).strip().upper()
        trade_symbol = str(scrip.get("pTrdSymbol", "")).strip().upper()
        token = scrip.get("pSymbol")
        if symbol in symbols and trade_symbol == f"{symbol}-EQ" and token:
            tokens_by_symbol[symbol] = str(token)

    missing_symbols = sorted(symbols - tokens_by_symbol.keys())
    if missing_symbols:
        raise RuntimeError(
            "Could not find NSE equity tokens for Nifty 100 symbols: "
            + ", ".join(missing_symbols)
        )

    tokens = [
        WsToken(exchange_segment="nse_cm", instrument_token=tokens_by_symbol[symbol])
        for symbol in sorted(symbols)
    ]
    tokens.append(WsToken(exchange_segment="nse_cm", instrument_token="256265"))
    return tokens


async def _sfeed_worker():
    """Async worker connecting to Kotak's binary WebSocket."""
    try:
        consumer_key = _required_env("NEO_CONSUMER_KEY")
        mobile_number = _required_env("NEO_MOBILE_NUMBER")
        ucc = _required_env("NEO_UCC")
        totp = _required_env("NEO_TOTP_CODE")
        mpin = _required_env("NEO_MPIN")

        client = NeoAPI(
            consumer_key=consumer_key,
            environment="prod"
        )
        
        login_response = client.totp_login(
            mobile_number=mobile_number,
            ucc=ucc,
            totp=totp
        )
        _require_auth_data(login_response, "TOTP login")
        
        validation_response = client.totp_validate(
            mpin=mpin
        )
        _require_auth_data(validation_response, "MPIN validation")
        
        print("✅ Kotak SFeed authenticated successfully")
        
        tokens_to_subscribe = _get_nifty_100_tokens(client)
        
        async with client.create_websocket() as ws:
            print(
                f"🔗 Kotak SFeed Connected. Subscribing to "
                f"{len(tokens_to_subscribe) - 1} Nifty 100 stocks and Nifty 50..."
            )
            await ws.subscribe_scrips(tokens_to_subscribe)
            
            async for message in ws:
                if isinstance(message, SFeedScrip):
                    symbol = message.trading_symbol 
                    ltp = message.last_traded_price
                    
                    if symbol and ltp:
                        clean_sym = symbol.replace("-EQ", "")
                        yfinance_sym = f"{clean_sym}.NS" if not clean_sym.startswith("NIFTY") else "^NSEI"
                        live_market_data[yfinance_sym] = float(ltp)
                        
    except Exception as e:
        print(f"WebSocket Connection Error: {e}")

def start_kotak_feed():
    """Spins up the WebSocket in a daemon thread so Streamlit doesn't block."""
    def run_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_sfeed_worker())
    
    t = threading.Thread(target=run_loop, daemon=True)
    t.start()
    return t