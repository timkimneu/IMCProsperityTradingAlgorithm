import string
import jsonpickle
import numpy as np
import enum
import math
import pandas as pd

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List,Any

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

AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"
CHOCOLATE = "CHOCOLATE"
STRAWBERRIES = "STRAWBERRIES"
ROSES = "ROSES"
GIFT_BASKET = "GIFT_BASKET"

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000,
    ORCHIDS : 11_000,
    CHOCOLATE: 0,
    STRAWBERRIES: 0,
    ROSES: 0,
    GIFT_BASKET: 0,
}

PRODUCTS = [
    AMETHYSTS,
    STARFRUIT,
    ORCHIDS,
    CHOCOLATE,
    STRAWBERRIES,
    ROSES,
    GIFT_BASKET
]

POSITION_LIMITS = {
    AMETHYSTS : 20,
    STARFRUIT : 20,
    ORCHIDS : 100,
    CHOCOLATE : 250,
    STRAWBERRIES : 350,
    ROSES : 60,
    GIFT_BASKET: 60
}

SUM_OF_PARTS = "SUM_OF_PARTS"

ROSES_GIFT_BASKET_SPREAD = "ROSES_GIFT_BASKET_SPREAD"
SUM_GIFT_BASKET_SPREAD = "SUM_GIFT_BASKET_SPREAD"

SPREADS = [
    ROSES_GIFT_BASKET_SPREAD, 
    SUM_GIFT_BASKET_SPREAD
    ]

SPREAD_VOL = 5

class STRATEGY_TYPE(enum.Enum):
    NAIVE = 0
    ROBUST = 1
    COMPLEX = 2

class Trader:

    def __init__(self):
        """
        Intializes the Trader class
        """
        self.window = 30

        self.price_history = dict()
        for product in PRODUCTS:
            self.price_history[product] = []

        self.pred_last_price = dict()
        for product in PRODUCTS:
            self.pred_last_price[product] = None
        self.pred_last_price[SUM_OF_PARTS] = None

        self.ema_prices = dict()
        for product in PRODUCTS:
            self.ema_prices[product] = None
        self.ema_prices[SUM_OF_PARTS] = None

        self.spread = dict()
        for spread in SPREADS:
            self.spread[spread] = []

        self.signals_history = dict()
        for spread in SPREADS:
            self.signals_history[spread] = []
        self.signals_history[STARFRUIT] = []

        self.signals_state = dict()
        for spread in SPREADS:
            self.signals_state[spread] = None # none means no signal
        self.signals_state[STARFRUIT] = None # True means signal to continue to buy x amount, False means continue to sell x amount
        # {PRODUCT: (True, 10)}

        self.stdev = dict() # To store the standard deviation
        for product in PRODUCTS:
            self.stdev[product] = None
        
        self.amethyst_strategy_type = STRATEGY_TYPE.ROBUST # test different strategies
        self.starfruit_strategy_type = STRATEGY_TYPE.ROBUST # test different strategies
        self.orchid_strategy_type = STRATEGY_TYPE.NAIVE
        self.gift_basket_sum_of_parts_strategy = STRATEGY_TYPE.NAIVE
        self.roses_gift_basket_strategy = STRATEGY_TYPE.NAIVE
        self.vol_threshold = 2 # our volatility threshold
        #self.pct_change_threshold = .1 # our percent change threshold
        self.ema_param = .5
    
    def get_position(self, product, state :TradingState) -> int:
        """Gets the current position size of a product. Returns 0 if no position."""
        return state.position.get(product, 0)
    
    def get_mid_price(self, product, state :TradingState) -> float:
        """Calculates the mid_price based on the average of the lowest ask and highest bid"""
        order_depth: OrderDepth = state.order_depths.get(product)
        if order_depth == None:
            return DEFAULT_PRICES[product]

        market_bids = order_depth.buy_orders
        if len(market_bids) == 0:
            # There are no bid orders in the market (midprice undefined)
            raise ValueError("No Bids" + product)
        
        market_asks = order_depth.sell_orders
        if len(market_asks) == 0:
            # There are no ask orders in the market (mid_price undefined)
            return ValueError("No Asks" + product)
        
        best_bid = max(market_bids)
        best_ask = min(market_asks)
        return (best_bid + best_ask)/2
    
    def update_stdev(self) -> None:
        if len(self.price_history.get(PRODUCTS[0])) >= self.window:
            for product in PRODUCTS:
                price_history = np.array(self.price_history[product][-self.window:])
                self.stdev[product] = np.std(price_history)
    

    def calculate_ema_prices(self, state :TradingState, product) -> None:
        """
        Returns the exponential moving average of the prices of each product.
        """
        mid_price = self.get_mid_price(product, state)
        # if the ema is not calculated yet (before rolling period has started), then set the ema to the mid_price
        if self.ema_prices[product] == None:
            return mid_price
        else:
            ema_param = .5
            return ema_param * mid_price + (1-ema_param) * self.ema_prices[product]

    def update_prices(self, state :TradingState) -> None:
        """Updates the prices for both ema_price and the predicted_price

        Args:
            state (TradingState): state of Trader
        """
        for product in PRODUCTS:
            self.ema_prices[product] = self.calculate_ema_prices(state, product)
            self.pred_last_price[product] = self.get_mid_price(product,state) # set the predicted price as the last price
            if not self.price_history[product]:
                self.price_history[product] = []
            self.price_history[product].append(self.get_mid_price(product,state))
        
        # update the sum_of_parts price
        self.pred_last_price[SUM_OF_PARTS] = 4.0 * self.get_mid_price(CHOCOLATE,state) + 6.0 * self.get_mid_price(STRAWBERRIES, state) + self.get_mid_price(ROSES,state)
    
    def update_spreads(self,state :TradingState) -> None:
        spread = self.pred_last_price[SUM_OF_PARTS] - self.pred_last_price[GIFT_BASKET]

        # update the spread
        self.spread[SUM_GIFT_BASKET_SPREAD].append(spread)


    def filter_orderbook(self,product,price:float,state:TradingState,buy_order:bool) -> dict[int,int]:
        """Filters all the orders that are above/below the price depending on if they are sell or buy orders."""
        order_depth: OrderDepth = state.order_depths[product]

        filtered_order_book = {}

        if buy_order:
            buy_orders = order_depth.buy_orders
            for bid_price in buy_orders.keys():
                # if the bid price is greater than the acceptable_price then sell to them
                if bid_price >= price:
                    filtered_order_book.update({bid_price:buy_orders[bid_price]})
        else:
            sell_orders = order_depth.sell_orders
            for ask_price in sell_orders.keys():
                # if the ask price is below the acceptable_price then buy from them
                if ask_price <= price:
                    filtered_order_book.update({ask_price:sell_orders[ask_price]})

        return filtered_order_book

    def fill_orders(self,order_book :dict[int,int], product, quantity :int, buy_orders :bool) -> tuple[int,List[Order]]:
        """
        Fills a dictionary of market orders in hopes of executing a trade. The order_book is assumed to be filtered to appropriate price.

        - order_book: dictionary of {price, quantity (+/-)} pairs 
        - product: the product
        - quantity: the abs(volume) of the trades to fill 
        - buy_order: if we are filling buy_orders (TRUE) or sell_orders (FALSE)

        Returns a quantity_left_over, List[Order(price, symbol, quantity)]
        """
        orders = []

        if buy_orders: 
            # filling buy orders
            if len(order_book) == 0:
                return quantity, orders 
            for order_price, order_quantity in order_book.items():        
                if order_quantity > quantity: 
                    # fill as much as possible
                    order_quantity = quantity
                    quantity = 0
                else:
                    quantity = quantity - order_quantity
                
                # create a sell order to match the buy order.
                orders.append(Order(product, order_price, -order_quantity))

                if quantity == 0:
                    break # break out of the loop
                elif quantity < 0:
                    raise ValueError("Miscalculated position sizes somehow" + str(order_book) + str(quantity))
            return quantity, orders
                
        else:
            # filling sell orders
            if len(order_book) == 0:
                return quantity, orders 
        
            for order_price, order_quantity in order_book.items():
                # sell orders have negative quantity, absolute value of the quantity
                order_quantity = abs(order_quantity)

                if order_quantity > quantity:
                    order_quantity = quantity
                    quantity = 0
                else:
                    quantity = quantity - order_quantity

                # create a buy order to match the sell order, quantity is positive 
                orders.append(Order(product, order_price, order_quantity))

                if quantity == 0:
                    break # break out of the loop
                elif quantity < 0:
                    raise ValueError("Miscalculated position sizes somehow" + str(order_book) + str(quantity))
            return quantity, orders

    def create_basket_order(self,state,buy:bool, volume:int) -> dict[PRODUCTS:List[Order]]:
        """Creates a basket order for Gift Baskets and it's contents where a gift basket = 4 * Chocolate + 6 * Bannanas + Roses. Trys to totally fill a basket order where there will only be x complete gift basket orders and no partial orders.

        Args:
            state (_type_): _description_
            buy (bool): buy = True, sell = False
            volume (int): the abs(total quantity) of buy/sell orders

        Returns:
            List[Order]: list of orders to execute
        """

        if buy:
            sign = 1 
        else:
            sign = -1
        results :dict[PRODUCTS:List[Order]] = dict()
        basket = {CHOCOLATE: 4, STRAWBERRIES : 6, ROSES : 1}
        total_fill = volume

        """
        # Adjust total_fill based on the maximum allowable for each component
        for product, quantity in basket.items():
            current_position = self.get_position(product, state)
            if buy:
                max_product_fill = (POSITION_LIMITS[product] - current_position) // quantity
            else:
                max_product_fill = (-POSITION_LIMITS[product] -current_position) // quantity

            total_fill = min(total_fill, abs(max_product_fill))

        if total_fill > 0:
            for product in basket.keys():
                bid_spread = 0
                ask_spread = 0
                if product == STRAWBERRIES:
                    bid_spread = 1.5 # TODO change to 1
                    ask_spread = 1
                elif product == ROSES:
                    bid_spread = 1 # TODO change to 1
                    ask_spread = 1.5
                elif product == CHOCOLATE:
                    bid_spread = 1.5
                    ask_spread = 1
                orders = []
                if buy:
                    # lowball
                    filtered_orders = self.filter_orderbook(product, math.floor(self.ema_prices[product] + bid_spread), state,not buy)
                    remainder, new_orders = self.fill_orders(filtered_orders, product,total_fill * basket[product], not buy)

                    orders.extend(new_orders)
                    
                    # post any left over orders
                    orders.append(Order(product, math.floor(self.ema_prices[product] + bid_spread), remainder * sign)) # since if sell, then the volume is negative
                else:
                    #lowball
                    filtered_orders = self.filter_orderbook(product, math.ceil(self.ema_prices[product] - ask_spread), state,not buy)
                    remainder, new_orders = self.fill_orders(filtered_orders, product,total_fill * basket[product],not buy)

                    orders.extend(new_orders)
                    
                    # post any left over orders
                    orders.append(Order(product, math.ceil(self.ema_prices[product] - ask_spread), remainder * sign)) # since if sell, then the volume is negative
                results.update({product:orders})
            """
            
        orders = []

        if buy:
            orders.append(Order(GIFT_BASKET, math.ceil(self.ema_prices[GIFT_BASKET]), (total_fill) * -sign)) # negative since theposition will be the inverse
        else:
            orders.append(Order(GIFT_BASKET, math.floor(self.ema_prices[GIFT_BASKET]), (total_fill) * -sign))

        results.update({GIFT_BASKET:orders})
        return results
         
    def gift_basket_sum_of_parts_pair_trade_naive(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:
        """
        """        
        spread = self.spread[SUM_GIFT_BASKET_SPREAD][-1]
        if len(self.spread[SUM_GIFT_BASKET_SPREAD]) < self.window:
            raise ValueError("Trading before end of window" + str(len(self.spread[SUM_GIFT_BASKET_SPREAD])) + str(self.window))

        spread_30_days = self.spread[SUM_GIFT_BASKET_SPREAD][-self.window:]
        spread_mean = np.mean(spread_30_days)
        spread_stdev = np.std(spread_30_days)
        z_score = (spread - spread_mean)/spread_stdev

        logger.print(f"Mean spread: {spread_mean}, Std: {spread_stdev}, Z-Score: {z_score} Current Spread: {spread}")

        results = {}
        buy = False
        sell = False
        last_signal = None
        if self.signals_history[SUM_GIFT_BASKET_SPREAD]:
            last_signal = self.signals_history[SUM_GIFT_BASKET_SPREAD][-1][1]
            last_timestep = self.signals_history[SUM_GIFT_BASKET_SPREAD][-1][0]
        
        if z_score < -2.0 and not spread > -350 or (last_signal and last_signal == 1 and state.timestamp - last_timestep <= 200):
            if z_score < -2.0 and not spread > -350:
                buy = True
                # this means that spread = SUM_OF_PARTS - GIFT_BASKET will mean revert and increase. Asset1 will increase and asset2 will decrease.
            # TODO Implement super signals
            #if spread < -450: #500?
                # super strong signal to buy
            #    results.update(self.create_basket_order(state,True, SPREAD_VOL * 5)) #buy 25
            #else:
            results.update(self.create_basket_order(state,True, 2*SPREAD_VOL))
                
        elif z_score > 2.0 or (last_signal and last_signal == -1 and state.timestamp - last_timestep <= 200):
            if z_score > 2.0:
                sell = True
            # this means that spread = SUM_OF_PARTS - GIFT_BASKET will mean revert and decrease. Asset1 will decrease and asset2 will increase.

            # TODO Implement Super Signals
            #if spread > -300:
                # super strong signal to sell
                #results.update(self.create_basket_order(state,False, SPREAD_VOL * 5)) # sell 25
            #else:
            results.update(self.create_basket_order(state,False, 2*SPREAD_VOL))

        if buy: 
            self.signals_history[SUM_GIFT_BASKET_SPREAD].append((state.timestamp, 1))
        elif sell:
            self.signals_history[SUM_GIFT_BASKET_SPREAD].append((state.timestamp, -1))

        return results
        

    def gift_basket_sum_of_parts_pair_trade(self, state :TradingState) -> dict[PRODUCTS,List[Order]]:
        """
        Performs pair trading based on the spread between SUM_OF_PARTS - GIFT BASKET price. 

        Args:
            state (TradingState): _description_
        """

        if self.gift_basket_sum_of_parts_strategy == STRATEGY_TYPE.NAIVE:
            order_dict = self.gift_basket_sum_of_parts_pair_trade_naive(state)
        else:
            raise ValueError("Strategy not implemented")
        return order_dict
    
    def roses_gift_basket_pair_trade_naive(state: TradingState) -> dict[PRODUCTS,List[Order]]:
        return

    def roses_gift_basket_pair_trade(self, state :TradingState) -> dict[PRODUCTS,List[Order]]:
        """_summary_

        Args:
            state (TradingState): _description_
        """

        if self.roses_gift_basket_strategy == STRATEGY_TYPE.NAIVE:
            order_dict = self.roses_gift_basket_pair_trade_naive(state)
        else:
            raise ValueError("Strategy not Implemented")
        return order_dict
    
    def amethyst_robust_strategy(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:
        """
        Arbitrage differences in bid and ask around 10,000, breaking down the orderbook to strategically place orders.
        """
        acceptable_price = DEFAULT_PRICES[AMETHYSTS]
        position_amethysts = self.get_position(AMETHYSTS, state)

        max_bid_volume = POSITION_LIMITS[AMETHYSTS] - position_amethysts 
        max_ask_volume = -POSITION_LIMITS[AMETHYSTS] - position_amethysts 

        orders = []

        # filter the orderbook for ask orders below acceptable price and buy orders above the acceptable price
        buy_orders = self.filter_orderbook(AMETHYSTS, acceptable_price, state, True)
        sell_orders = self.filter_orderbook(AMETHYSTS, acceptable_price, state, False)

        # fill the acceptable orders up to the max_sell_volume for buy orders
        ask_quantity_remainder, filled_orders = self.fill_orders(buy_orders, AMETHYSTS, abs(max_ask_volume), True)
        orders += filled_orders

        # fill the acceptable orders up to the max_buy_volume for sell orders
        buy_quantity_remainder, filled_orders = self.fill_orders(sell_orders, AMETHYSTS, abs(max_bid_volume), False)
        orders += filled_orders

        orders.append(Order(AMETHYSTS, math.floor(self.ema_prices[AMETHYSTS] - 1), buy_quantity_remainder))
        orders.append(Order(AMETHYSTS, math.ceil(self.ema_prices[AMETHYSTS] + 1), -ask_quantity_remainder))

        return {AMETHYSTS:orders}

    def amethyst_strategy(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:
        """
        - Naive Strategy is to post max order sizes at sell price +1 and buy price -1. Market make around 10000. 
        - Robust Strategy is fill any orders below/above 10000 (market taking). Then with the leftover quantity, market make around 10000. 
        """

        #if self.amethyst_strategy_type == STRATEGY_TYPE.NAIVE:
        #    orders = self.amethyst_naive_strategy(state)
        if self.amethyst_strategy_type == STRATEGY_TYPE.ROBUST:
            order_dict = self.amethyst_robust_strategy(state)
        elif self.amethyst_strategy_type == STRATEGY_TYPE.COMPLEX:
            # TODO 
            print()
        return order_dict

    def starfruit_robust_strategy(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:

        position_starfruit = self.get_position(STARFRUIT, state)

        max_bid_volume = POSITION_LIMITS[STARFRUIT] - position_starfruit
        max_ask_volume = -POSITION_LIMITS[STARFRUIT] - position_starfruit
        orders = []

        #if volatility <= self.vol_threshold:

        # filter the orderbook for ask orders below acceptable price and buy orders above the acceptable price
        buy_orders = self.filter_orderbook(STARFRUIT, self.ema_prices[STARFRUIT], state, True)
        sell_orders = self.filter_orderbook(STARFRUIT, self.ema_prices[STARFRUIT], state, False)

        # fill the acceptable orders up to the max_sell_volume for buy orders
        ask_quantity_remainder, filled_orders = self.fill_orders(buy_orders, STARFRUIT, abs(max_ask_volume), True)
        orders += filled_orders

        # fill the acceptable orders up to the max_buy_volume for sell orders
        buy_quantity_remainder, filled_orders = self.fill_orders(sell_orders, STARFRUIT, abs(max_bid_volume), False)
        orders += filled_orders

        orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 1), buy_quantity_remainder))
        orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 1), -ask_quantity_remainder))

        """
        if position_starfruit == 0:
            # Not long nor short
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 1), buy_quantity_remainder))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 1), -ask_quantity_remainder))
        elif position_starfruit > 0:
            # Long position
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] - 2), buy_quantity_remainder))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT]), -ask_quantity_remainder))

        else:
            # Short position
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT]), buy_quantity_remainder))
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] + 2), -ask_quantity_remainder))
        """
        
        return {STARFRUIT:orders}        
    
    def starfruit_complex_strategy(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:
        """Idea: Use rolling Z-score"""

        if len(self.price_history[STARFRUIT]) < self.window:
            raise ValueError("Trading before end of window" + str(self.window))

        data_30_days = self.price_history[STARFRUIT][-self.window:]
        mean = np.mean(data_30_days)
        stdev = np.std(data_30_days)
        z_score = (self.ema_prices[STARFRUIT] - mean)/stdev

        buy = False
        sell = False

        last_signal = None
        if self.signals_history[STARFRUIT]:
            last_signal = self.signals_history[STARFRUIT][-1][1]
            last_timestep = self.signals_history[STARFRUIT][-1][0]

        logger.print(f"Mean: {mean}, Std: {stdev}, Z-Score: {z_score} Current Price: {self.ema_prices[STARFRUIT]}")
        max_bid_volume = POSITION_LIMITS[STARFRUIT] - self.get_position(STARFRUIT,state)
        max_ask_volume = -POSITION_LIMITS[STARFRUIT] - self.get_position(STARFRUIT,state)
        orders = []
        if z_score < -2 or (last_signal and last_signal == 1 and state.timestamp - last_timestep <= 300):
            buy = True
            # slightly overpay
            orders.append(Order(STARFRUIT, math.floor(self.ema_prices[STARFRUIT] + 1), min(max_bid_volume, SPREAD_VOL)))
        elif z_score > 2 or (last_signal and last_signal == -1 and state.timestamp - last_timestep <= 300):
            sell = True
            # slightly undersell
            orders.append(Order(STARFRUIT, math.ceil(self.ema_prices[STARFRUIT] - 1), max(max_ask_volume, -SPREAD_VOL)))

        if buy: 
            self.signals_history[STARFRUIT].append((state.timestamp, 1))
        elif sell:
            self.signals_history[STARFRUIT].append((state.timestamp, -1))

        return {STARFRUIT:orders}


    def starfruit_strategy(self, state :TradingState) -> dict[PRODUCTS:List[Order]]:
        """
        Trade based on the mid_price estimate of an exponential moving average of the rolling period. 
        - Naive Strategy is buy/sell to the max position limit at the ema_price + 1/-1 unless the volatility of prices is above +1.5/-1.5 (which suggests price corrections). At which point we then trade on the linear trend

        """
        #if self.starfruit_strategy_type == STRATEGY_TYPE.NAIVE:
        #    orders = self.starfruit_naive_strategy(state)
        if self.starfruit_strategy_type == STRATEGY_TYPE.ROBUST:
            order_dict = self.starfruit_robust_strategy(state)
        elif self.starfruit_strategy_type == STRATEGY_TYPE.COMPLEX:
            order_dict = self.starfruit_complex_strategy(state)
        return order_dict

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        #logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        result = {}
        self.update_prices(state)
        self.update_stdev()
        self.update_spreads(state)
        logger.print("EMA Prices: " + str(self.ema_prices) + "Last Predicted Prices: " + str(self.pred_last_price))
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            order_dict: dict[PRODUCTS:List[Order]] = dict()
            orders: List[Order] = []

            logger.print("------------|")
            logger.print(product + "|")
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", " + str(order_depth.buy_orders) + ", Sell order depth : " + str(len(order_depth.sell_orders)) + ", " + str(order_depth.sell_orders) + "|")
            logger.print("Own Trades: " + str(state.own_trades.get(product, None)) + "|")
            logger.print("Market Trades: " + str(state.market_trades.get(product, None)) + "|")
            logger.print("Own Positions: " + str(state.position.get(product, None)) + "|")
            
            if product == GIFT_BASKET:
                if self.price_history[product] and len(self.price_history[product]) >= self.window:
                    # start the strategy
                    order_dict = self.gift_basket_sum_of_parts_pair_trade(state)
                    logger.print("Signals: " + str(self.signals_history[SUM_GIFT_BASKET_SPREAD]))
            elif product == AMETHYSTS:
                order_dict = self.amethyst_strategy(state)
            elif product == STARFRUIT:
                if self.price_history[product] and len(self.price_history[product]) >= self.window:
                    order_dict = self.starfruit_strategy(state)
                    logger.print("Signals: " + str(self.signals_history[product]))

            logger.print("Own Orders: " + str(orders))
            
            #result[product] = orders
            result.update(order_dict)
            
        # serialize the previous mid_prices for STARFRUIT and store those values
        traderData = "" # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.

        conversions = 0
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

    """
    def serialize_state(self, state: TradingState) -> str:
        # Deserialize the traderData from the last iteration
        if state.traderData and state.traderData.strip():
            self.price_history = jsonpickle.decode(state.traderData)
        else:
            self.price_history = dict()
            for product in PRODUCTS:
                self.price_history[product] = []
            
        for product in PRODUCTS:
            # Update the price history with new data from the current iteration
            mid_price = self.get_mid_price(product, state)
            self.price_history[product].append(mid_price)
            if len(self.price_history[product]) > self.window:
                # Keep only the last 30 days of prices, for example
                self.price_history[product] = self.price_history[product][-self.window:]

        # Serialize the updated price history for the next iteration
        return jsonpickle.encode(self.price_history)
    """