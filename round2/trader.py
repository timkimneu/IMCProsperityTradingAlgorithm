from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId
from typing import List, Any
import string
import jsonpickle
import json
import numpy as np
import enum
import math

AMETHYSTS = "AMETHYSTS"
STARFRUIT = "STARFRUIT"
ORCHIDS = "ORCHIDS"

OPTIMAL_SUNLIGHT = 7 * 360
# any dip below 7 hours with 4% production dip for 10 minutes
SUNLIGHT_RATE_OF_CHANGE = .04/10

OPTIMAL_HUMIDITY = [60,80]
# 2% dip in production for 5% humidity change outside optimal range
HUMIDTY_RATE_OF_CHANGE = .4 # note humidity is already in percentage (%)


PRODUCTS = [
    AMETHYSTS,
   STARFRUIT,
   ORCHIDS,
]

POSITION_LIMITS = {
    ORCHIDS : 100,
    AMETHYSTS : 20,
    STARFRUIT : 20
}

DEFAULT_PRICES = {
    AMETHYSTS : 10_000,
    STARFRUIT : 5_000, # starfruit price will change
    # TODO orchid?
}

class STRATEGY_TYPE(enum.Enum):
    NAIVE = 0
    ROBUST = 1
    COMPLEX = 2


class Trader:

    def __init__(self):
        self.rolling_window = 30 # set the rolling window as 30
        self.ema_rolling_window = 2 # set the ema rolling window as 2

        self.price_history = dict()  # To store the last self.rolling_window timesteps of price data
        for product in PRODUCTS:
            self.price_history[product] = []

        self.ema_prices = dict() # To store the predicted EMA prices 
        for product in PRODUCTS:
            self.ema_prices[product] = None

        self.stdev_hist = dict() # To store the standard deviation
        for product in PRODUCTS:
            self.stdev_hist[product] = []

        self.pct_change = dict() # To store the pct_change
        for product in PRODUCTS:
            self.pct_change[product] = []

        self.amethyst_strategy_type = STRATEGY_TYPE.ROBUST # test different strategies
        self.starfruit_strategy_type = STRATEGY_TYPE.ROBUST # test different strategies
        self.orchid_strategy_type = STRATEGY_TYPE.NAIVE
        self.vol_threshold = 2 # our volatility threshold
        self.pct_change_threshold = .1 # our percent change threshold

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
        return (best_bid + best_ask)/2
    
    def get_position(self, product, state :TradingState) -> int:
        """Gets the current position size of a product. Returns 0 if no position."""
        return state.position.get(product, 0)
    
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
        
    def update_ema_prices(self, state :TradingState) -> None:
        """
        Update the exponential moving average of the prices of each product.
        """
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)

            # if the ema is not calculated yet (before rolling period has started), then set the ema to the mid_price
            if self.ema_prices[product] == None:
                self.ema_prices[product] = mid_price
            else:
                #ema_param = (2 / (self.rolling_window + 1)) # calculate the ema_param #TODO could mess with this value
                ema_param = (2 / (self.ema_rolling_window + 1))
                self.ema_prices[product] = ema_param * mid_price + (1-ema_param) * self.ema_prices[product]
        return



    def naive_orchid_strategy(self, state :TradingState):
        """
        Extremely simple implementation, trade on the price of orchids based on an estimate moving average with window of 2. 
        Trade +/-2 below and above.
        Implement position limit at 1/2 the max of 100.

        If there is a large % change from the previous position, then update the estimate to be lower -1. (lower volume, increase spread)
        If the rolling standard deviation > 2, then in a period of high volatility trade directionally. (lower volume, increase spread)

        If both large % change and high volatility, increase volume and spread
        """
        return

    def orchid_strategy(self, state :TradingState):
       
        return

        
    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10;  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
    
        traderData = "SAMPLE" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 0
        return result, conversions, traderData