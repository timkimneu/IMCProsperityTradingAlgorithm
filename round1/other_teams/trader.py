from datamodel import *
from typing import List
import string
import numpy as np
import json
from typing import Any

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import math

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

    # definite init state
    def __init__(self):
        self.orders = {}
        self.conversions = 0
        self.traderData = "SAMPLE"
        
        
        # DATA:
        self.remaining_shares = {"AMETHYSTS" : 15, "STARFRUIT" : 20} # not useful anymore
        self.prev_price = {}
        self.prev_mid_price = {}
        self.product = 'STARFRUIT'

        # Following are used for Policy Calculation

        # used to normalize the data
        self.data_stats = {
                                    'spread'              :  (5.384692346173087, 1.897679366983876),
                                     'order_imbalance'    :  (35.26363181590796, 14.512051290191796),
                                     'smart_price'        :  (4980.373449615514, 14.399206920669522),
                                     'trade_sign'         :  (3.1515757878939468, 4.166280614924584),
                                     'order_volume'       :  (0.02301150575287644, 3.3452445944600897),
                                     'transaction_volume' :  (1.1570785392696348, 0.409275183110554),
                                     'price_move'         :  (-0.02351175587793897, 1.9767093493623005),
                                     'mid_price'          :  (2.6923461730865434, 0.94883968349193),
                                     }
        
        self.std_of_data = {}

        self.q_table = '00000000000000000000000000000000000000100000000000000000000000100001000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000011001100001000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000010100000000000000000000000010100000000000000000000000000000000000000000000000000000000000000000000000010100000000000000000000100110000000100000000000000001000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010100001000000000000000000010100000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000101011010010000000000000000111001010011100000000001111000010000000000000000000000000000000000000000000000000000000000000000001010110101000000000000000111110110010101000000000010110000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'

        self.current_sells = []
        self.current_buys = []
        self.trades = 0


    def send_sell_order(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, price, amount))

        if msg is not None:
            logger.print(msg)

    def send_buy_order(self, product, price, amount, msg=None):
        self.orders[product].append(Order(product, price, amount))

        if msg is not None:
            logger.print(msg)

    def printStuff(self, state):
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))        

    def calculate_spread_and_order_imbalance(self, state):
        # Calculates SPREAD
        # RETURNS NONE iff there is no sell or no buy orders
        # RETURNS dict {product : spread}, dict {product : order imbalance}
        spreads = {}
        order_imbalance = {}
        for product in state.order_depths:
            order_book = state.order_depths[product]
            sell_orders = order_book.sell_orders
            buy_orders = order_book.buy_orders

            if len(sell_orders) != 0 and len(buy_orders) != 0:
                ask, ask_volume = list(sell_orders.items())[0] # best ask
                bid, bid_volume = list(buy_orders.items())[0]  # best bid                
                spreads[product] = ask-bid
                order_imbalance[product] = bid_volume-ask_volume    
            else:
                spreads[product] = None
                order_imbalance[product] = None           
            
        return spreads, order_imbalance
    
    def calculate_mid_price(self, state):
        mid_price = {}
        for product in state.order_depths:
            order_book = state.order_depths[product]
            sell_orders = order_book.sell_orders
            buy_orders = order_book.buy_orders

            if len(sell_orders) != 0 and len(buy_orders) != 0:
                ask, ask_volume = list(sell_orders.items())[0] # best ask
                bid, bid_volume = list(buy_orders.items())[0]  # best bid   
                mid_price[product] = (ask-bid)/2             
            else:
                mid_price[product] = None
            
        return mid_price

    def calculate_transaction_volume(self, state):
        # Signed Transaction Volume: A signed quantity indicating the number of shares bought in the
        # last state minus the number of shares sold in the last state
        transaction_volume = {}
        for product in state.market_trades:
            market_trades = state.market_trades[product]
            transaction_volume[product] = len(market_trades)

        return transaction_volume
    
    def calculate_order_volume(self, state):
        # signed, represents order volume
        order_volume = {}
        for product in state.order_depths:
            order_book = state.order_depths[product]
            sell_orders = order_book.sell_orders
            buy_orders = order_book.buy_orders

            volume = 0
            if len(sell_orders) != 0:
                for _, amount in list(sell_orders.items()):
                    volume += amount
            if len(buy_orders) != 0:
                for _, amount in list(buy_orders.items()):
                    volume += amount
            
            order_volume[product] = volume

        return order_volume
    
    def calculate_immediate_order_cost(self, state):
        # Returns the cost we would pay for purchasing our 
        # remaining shares immediately with a market order.
        # returns NONE if we cannot fully fill our remaining shares right now
        immediate_order_cost = {}
        for product in state.order_depths:
            order_book = state.order_depths[product]
            sell_orders = order_book.sell_orders

            remaining = self.remaining_shares[product]
        
            if len(sell_orders == 0): 
                immediate_order_cost[product] = None
            else: 
                orders = list(sell_orders.items())
                cost = 0
                while remaining > 0 and len(orders) > 0:
                    ask, amount = orders

                    if remaining + amount <= 0:
                        cost += ask * remaining
                        immediate_order_cost[product] = cost

                    elif remaining + amount > 0:
                        cost += ask * amount
                        remaining += amount
                
                if remaining > 0:
                    immediate_order_cost[product] = None
        
        return immediate_order_cost

    def calculate_smart_price(self, state):
        smart_price = {}
        for product in state.order_depths:
            order_book = state.order_depths[product]
            sell_orders = order_book.sell_orders
            buy_orders = order_book.buy_orders

            if len(sell_orders) != 0 and len(buy_orders) != 0:
                ask, ask_volume = list(sell_orders.items())[0] # best ask
                bid, bid_volume = list(buy_orders.items())[0]  # best bid        
                
                ask_volume = abs(ask_volume) # make non negative

                smart_price[product] = ((1/bid_volume) * ask + (1/ask_volume) * bid) / (1/bid_volume + 1/ask_volume)
            else:
                smart_price[product] = None
            
        return smart_price

    def calculate_price_move(self, state):
        price_move = {}
        for product in state.market_trades:
            market_trades = state.market_trades[product]

            volume = 0
            price = 0
            for trade in market_trades:
                volume += trade.quantity
                price += trade.price * trade.quantity
            
            if volume == 0: 
                price_move[product] = 0
            else:
                weighted_price = price / volume
                # set previous price to current price if its none
                prev_price = self.prev_price.get(product, weighted_price)

                # update price (signed)
                price_move[product] = weighted_price - prev_price

                # update prev price
                self.prev_price[product] = weighted_price
        
        return price_move

    def calculate_trade_sign(self, state):
        trade_sign = {}
        for product in state.market_trades:
            market_trades = state.market_trades[product]

            # set to current mid_price if not available.
            prev_mid_price = self.prev_mid_price.get(product, None)

            if prev_mid_price is None:
                self.prev_mid_price = self.calculate_mid_price(state)
                prev_mid_price = self.prev_mid_price[product]

            sign = 0
            for trade in market_trades:
                if trade.price > prev_mid_price:
                    sign += trade.quantity
                elif trade.price < prev_mid_price:
                    sign -= trade.quantity
            
            trade_sign[product] = sign

        # update prev mid price!
        self.prev_mid_price = self.calculate_mid_price(state)
        return trade_sign

    def collect_data(self, state):
        spreads, order_imbalance = self.calculate_spread_and_order_imbalance(state)
        smart_price = self.calculate_smart_price(state)
        trade_sign = self.calculate_trade_sign(state)
        price_move = self.calculate_price_move(state)
        mid_price = self.calculate_mid_price(state)

        return [('spread' ,spreads), 
                ('order_imbalance' ,order_imbalance),
                ('smart_price' ,smart_price), 
                ('trade_sign' ,trade_sign),
                ('price_move' ,price_move),
                ('mid_price' , mid_price)]

    def find_bin(self, value):
        bin_edges = [-2, -1, 1, 2, float('inf')]
        for i, edge in enumerate(bin_edges):
            if value < edge:
                return i
        return 1

    def data_to_state(self, data):
        res = []
        for feature, d in data:
            mean, std = self.data_stats[feature]
            value = d[self.product]

            if value is None: return None
            normalized = (value-mean)/std
            bin = self.find_bin(normalized)
            res.append(bin)
    
        return res
        
    def get_state_index(self, data):
        state = self.data_to_state(data)
        if state is None:
            return None
        v1, v2, v3, v4, v5 = state      
        # Adjust the values to be 0-based and compute the base-5 index
        index = ((((v1) * 5  +(v2)) * 5 + (v3)) * 5 + (v4)) * 5 + (v5)
        return index
    
    def get_action(self, data):
        index = self.get_state_index((data))
        if index is None: 
            return None
        else: 
            return self.q_table[index]

    def run(self, state: TradingState):        
        self.orders = {"STARFRUIT" : [], 'AMETHYSTS': []}
        self.conversions = 0
        
        spreads, order_imbalance, smart_price, trade_sign, price_move, mid_prices = self.collect_data(state)
        mid_price = mid_prices[1][self.product]
        data = [spreads, order_imbalance, smart_price, trade_sign, price_move]
        action = self.get_action(data)
        
        self.close_positions(state, mid_price)

        if action is None:
            self.traderData = 'NONE in Data'
        else:
            self.check_if_filled(state)

            # Sell
            if action == '1':
                self.send_sell_order(self.product, math.ceil(mid_price), -1)
                self.traderData = 'SELL'
            # Buy
            else:
                self.send_buy_order(self.product, math.floor(mid_price), 1)
                self.traderData = 'BUY'

        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData