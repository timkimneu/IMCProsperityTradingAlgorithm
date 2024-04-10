from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
import string
import jsonpickle
import json
import numpy as np


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()


class Trader:

    def __init__(self):
        """
        Intializes the Trader class
        """
        self.price_history = []  # To store the last 30 timesteps of prices
        self.position_limit = 20 # set the position limit as 20 for the tutorial/round1 TODO Remember to change this for each round

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            logger.print(product)
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", " + str(order_depth.buy_orders) + ", Sell order depth : " + str(len(order_depth.sell_orders)) + ", " + str(order_depth.buy_orders))
            logger.print("Own Trades: " + str(state.own_trades))
            logger.print("Market Trades: " + str(state.market_trades))
            logger.print("Own Positions: " + str(state.position))
            logger.print()

            if product == "AMETHYSTS":
                acceptable_price = 10000 # market make around 10k. #TODO position sizing
            elif product == "STARFRUIT":
                if len(self.price_history) >= 30:
                    price_history = np.array(self.price_history) # Participant should calculate this value
                    acceptable_price = np.average(price_history) # mean-reversion strategy
                    # implement stop-loss/take-profit and standard deviation or % drift
                else:
                    acceptable_price = 5000 # set to default of 5000

            logger.print("Acceptable price : " + str(acceptable_price))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                #TODO fix the position size checks
                # check if the amount would go over the position limit, if yes, then fill up to position limit
                #if abs(-best_ask_amount + state.position[product]) > self.position_limit:
                #    best_ask_amount = self.position_limit - abs(state.position[product])
                if int(best_ask) < acceptable_price:
                    logger.print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                #TODO fix the position size checks
                # check if the amount would go over the position limit, if yes, then fill up to position limit
                #if abs(best_bid_amount + state.position[product]) > self.position_limit:
                #    best_bid_amount = self.position_limit - abs(state.position[product])
                if int(best_bid) > acceptable_price:
                    logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            logger.print("Own Orders: " + str(orders))

            result[product] = orders
            
            # if it is starfruit
            if product == "STARFRUIT":
                # calculate the mid_price
                price = (best_ask + best_bid) / 2
        
        # serialize the previous mid_prices for STARFRUIT and store those values
        traderData = self.serialize_state(state,price) # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    def serialize_state(self, state: TradingState, curr_price :int) -> str:
        # Deserialize the traderData from the last iteration
        if state.traderData and state.traderData.strip():
            self.price_history = jsonpickle.decode(state.traderData)
        else:
            self.price_history = []

        # Update the price history with new data from the current iteration
        # Assuming you have a method to fetch or calculate the current price
        self.price_history.append(curr_price)
        if len(self.price_history) > 30:
            # Keep only the last 30 days of prices, for example
            self.price_history = self.price_history[-30:]

        # Serialize the updated price history for the next iteration
        return jsonpickle.encode(self.price_history)
    
# Trading Strategies
# def moving_avg(timeframe :int, state: TradingState) -> int:
        """
        Calculates the moving average based on the the past x days.
        - timeframe : past x days
        - state : current trading state

        returns the predicted price
        """