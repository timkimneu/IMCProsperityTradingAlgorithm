from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
import string
import jsonpickle
import json
import numpy as np
from enum import enum
import math

# constants
AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"

PRODUCTS = [
    AMETHYSTS,
   STARFRUIT,
]

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000, # starfruit price will change
}

class STRATEGY_TYPE(enum):
    NAIVE = 0
    ROBUST = 1
    COMPLEX = 2

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
        self.rolling_window = 30 # set the rolling window as 30

        self.price_history = dict()  # To store the last self.rolling_window timesteps of price data
        for product in PRODUCTS:
            self.price_history[product] = []

        self.ema_prices = dict() # To store the predicted EMA prices 
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.position_limit = 20 # set the position limit as 20 for the tutorial/round1 TODO Remember to change this for each round
        self.cash = 0 # start with 0 cash
        self.amethyst_strategy = STRATEGY_TYPE.NAIVE # test different strategies
        self.starfruit_strategy = STRATEGY_TYPE.NAIVE # test different strategies
        self.vol_threshold = 1.5 # our volatility strategy
        

    def get_mid_price(self, product, state :TradingState) -> float:
        """Calculates the mid_price based on the average of the lowest ask and highest bid"""
        order_depth: OrderDepth = state.order_depths[product]

        market_bids = order_depth.buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            return DEFAULT_PRICES[product]
        
        market_asks = order_depth.sell_orders
        if len(market_asks) == 0:
            # There are no ask orders in the market (mid_price undefined)
            return DEFAULT_PRICES[product]
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        logger.print(str((best_bid + best_ask)/2))
        return (best_bid + best_ask)/2
    
    def get_position(self, product, state :TradingState) -> int:
        """Gets the current position size of a product. Returns 0 if no position."""
        return state.position.get(product, 0)
    
    def get_value_on_product(self, product, state : TradingState):
        """
        Returns the amount of MONEY currently held on the product.  
        """
        return self.get_position(product, state) * self.get_mid_price(product, state)
    
    #TODO
    def update_pnl(self, state :TradingState) -> None:
        #TODO
        return 

    def calculate_volume(self, product, state :TradingState) -> tuple[int,int]:
        """
        Calculates the total volume of buy and sell orders for this product in the current state of the orderbook. Returns (buy volume, sell volume) where sell volume is positive.
        """
        order_depth: OrderDepth = state.order_depths[product]

        # buy volume
        buy_orders = list(order_depth.buy_orders.items())
        buy_vol = 0
        for idx in range(0,len(sell_orders)):
            bid_price, bid_amount = buy_orders[idx]
            buy_vol += bid_amount

        # sell volume
        sell_orders = list(order_depth.sell_orders.items())
        sell_vol = 0
        for idx in range(0,len(sell_orders)):
            ask_price, ask_amount = sell_orders[idx]
            sell_vol += ask_amount

        return (buy_vol,sell_vol)
    
    def filter_orderbook(self,product,price:float,state:TradingState,buy_order:bool) -> dict[int,int]:
        """Filters all the orders that are above/below the price depending on if they are sell or buy orders."""
        order_depth: OrderDepth = state.order_depths[product]

        filtered_order_book = {}

        if buy_order:
            buy_orders = order_depth.buy_orders.items()
            for idx in range(0,len(buy_orders)):
                bid_price, bid_amount = buy_orders[idx]
                if bid_price >= price:
                    filtered_order_book.update(buy_orders[idx])
        else:
            sell_orders = order_depth.sell_orders.items()
            for idx in range(0,len(sell_orders)):
                ask_price, ask_amount = sell_orders[idx]
                if ask_price <= price:
                    filtered_order_book.update(sell_orders[idx])

        return filtered_order_book
    
    def fill_orders(order_book :dict[int,int], product) -> List[Order]:
        """
        Fills a dictionary of market orders in hopes of executing a trade.

        - order_book: dictionary of {price, quantity (+/-)} pairs 
        - product: the product

        Returns a List[Order(price, symbol, quantity)]
        """
        return
    
    def calculate_volatility(self, product) -> float:
        """
        Calculates the volatility of a product from its price data over the rolling period of self.rolling_window
        - #TODO use the returns volatility?
        """
        price_history = np.array(self.price_history[product])
        #returns = np.diff(price_history) / price_history[:-1] 
        std_dev = np.std(price_history)
        return std_dev
    
    def calculate_position_sizing(self, product, state :TradingState) -> int:
        """Computes position sizing for a given product based on volatility"""
        return 
    
    def update_ema_prices(self, product, state :TradingState) -> None:
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)

            # if the ema is not calculated yet (before rolling period has started), then set the ema to the mid_price
            if self.ema_prices[product] == None:
                self.ema_prices[product] = mid_price
            else:
                ema_param = (2 / (self.rolling_window + 1)) # calculate the ema_param #TODO could mess with this value
                self.ema_prices[product] = ema_param * mid_price + (1-ema_param) * self.ema_prices[product]
        return
    
    def amethyst_naive_strategy(self, state :TradingState) -> List[Order]:
        position_amethysts = self.get_position(AMETHYSTS, state)

        bid_volume = self.position_limit - position_amethysts 
        ask_volume = - self.position_limit - position_amethysts 

        orders = []
        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] - 2, bid_volume))
        orders.append(Order(AMETHYSTS, DEFAULT_PRICES[AMETHYSTS] + 2, ask_volume))
        return orders

    def amethyst_strategy(self, state :TradingState) -> None:
        """
        Market make around 10000. 
        - Naive Strategy is to post max order sizes at sell price +2 and buy price -2
        - Robust Strategy is to buy/sell +2 and -2 and calculating posiiton size based on volume/current market orders
        """

        if self.amethyst_strategy == STRATEGY_TYPE.NAIVE:
            orders = self.amethyst_naive_strategy(state)
        elif self.amethyst_strategy == STRATEGY_TYPE.ROBUST:
            #TODO
            print()
        elif self.amethyst_strategy == STRATEGY_TYPE.COMPLEX:
            # TODO 
            print()
        return orders 
    
    def starfruit_naive_strategy(self, state :TradingState):

        def calculate_linear_reg(self, product):
            """Runs a least squares linear regression through the price data in the self.rolling_windows timeframe"""
            y_vals = np.array(self.price_history[product])
            x_vals = np.arange(0,self.rolling_window)
            x_mean = np.mean(x_vals)
            y_mean = np.mean(y_vals)
            slope = np.sum((x_vals - x_mean) * (y_vals - y_mean)) / np.sum((x_vals - x_mean)**2)
            b = y_mean - slope * x_mean

            return slope, b

        volatility = self.calculate_volatility(STARFRUIT)

        position_starfruit = self.get_position(STARFRUIT, state)

        bid_volume = self.position_limit - position_starfruit
        ask_volume = - self.position_limit - position_starfruit


        if volatility <= self.vol_threshold:
            orders = []

            # market make around the ema
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 1), bid_volume))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 1), ask_volume))
            
            return orders
        
        else: # if the volatility is high, then predict the price to increase/decrease directionally following a linear trend

            slope, _ = calculate_linear_reg(STARFRUIT)
            position_starfruit = self.get_position(STARFRUIT, state)

            if slope > 0:
                # we predict the mid price will continue to go up and that there is likely more mispricing
                # we want to hold an overall long position. This means that in expectation that the price will go up, we place a bid
                # slightly higher than expected. 
                orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT]), bid_volume))
                # we want to sell at a higher price, so we set the ask price higher
                orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] + 2), ask_volume))

            else:
                # we predict that the mid price will continue to go down and that there is likely more mispricing
                # we want to hold an overall short position. We will place a bid slightly lower than expected at a 1/2 the volume
                orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 2), math.floor(bid_volume / 2)))
                # we want to liquidate our position until the trend reverses, so our ask volume will still be the same.
                orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT]), ask_volume))

    
    def starfruit_strategy(self, state :TradingState) -> None:
        """
        Trade based on the mid_price estimate of an exponential moving average of the rolling period. 
        - Naive Strategy is buy/sell to the max position limit at the ema_price + 1/-1 unless the volatility of prices is above +1.5/-1.5 (which suggests price corrections). At which point we then trade on the linear trend

        """
        if self.starfruit_strategy == STRATEGY_TYPE.NAIVE:
            orders = self.starfruit_naive_strategy(state)
        elif self.starfruit_strategy == STRATEGY_TYPE.ROBUST:
            #TODO
            print()
        elif self.starfruit_strategy == STRATEGY_TYPE.COMPLEX:
            # TODO 
            print()
        return orders
    
    def serialize_state(self, state: TradingState) -> str:
        # Deserialize the traderData from the last iteration
        if state.traderData and state.traderData.strip():
            self.price_history = jsonpickle.decode(state.traderData)
        else:
            raise ValueError("No TraderData")
            
        for product in PRODUCTS:
            # Update the price history with new data from the current iteration
            mid_price = self.get_mid_price(product, state)
            self.price_history[product].append(mid_price)
            if len(self.price_history[product]) > self.rolling_window:
                # Keep only the last 30 days of prices, for example
                self.price_history[product] = self.price_history[product][-self.rolling_window:]

        # Serialize the updated price history for the next iteration
        return jsonpickle.encode(self.price_history)

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
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", " + str(order_depth.buy_orders) + ", Sell order depth : " + str(len(order_depth.sell_orders)) + ", " + str(order_depth.sell_orders))
            logger.print("Own Trades: " + str(state.own_trades))
            logger.print("Market Trades: " + str(state.market_trades))
            logger.print("Own Positions: " + str(state.position))
            logger.print()

            if product == "AMETHYSTS":
                acceptable_price = PRODUCTS[0] # market make around 10k. 
                logger.print("Acceptable price : " + acceptable_price)
                orders = self.amethyst_strategy(state) #TODO position sizing for other strategies

            elif product == "STARFRUIT":
                 if len(self.price_history) >= self.rolling_window:
                    logger.print("Acceptable price : " + str(self.ema_prices))
                    orders = self.starfruit_strategy(state) #TODO position sizing for other strategies

            logger.print("Own Orders: " + str(orders))

            result[product] = orders
            
        # serialize the previous mid_prices for STARFRUIT and store those values
        traderData = self.serialize_state(state) # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.

        conversions = 1 # not needed for round1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

"""
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
"""