from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np


class Trader:

    def __init__(self):
        """
        Intializes the Trader class
        """
        self.price_history = []  # To store the last 30 timesteps of prices
        self.position_limit = 20 # set the position limit as 20 for the tutorial TODO Remember to change this for each round

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            print(product)
            print()
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", " + str(order_depth.buy_orders) + ", Sell order depth : " + str(len(order_depth.sell_orders)) + ", " + str(order_depth.buy_orders))
            print("Own Trades: " + str(state.own_trades))
            print("Market Trades: " + str(state.market_trades))
            print("Own Positions: " + str(state.position))

            if product == "AMETHYSTS":
                acceptable_price = 10000
            elif product == "STARFRUIT":
                if len(self.price_history) >= 30:
                    price_history = np.array(self.price_history) # Participant should calculate this value
                    acceptable_price = np.average(price_history) # mean-reversion strategy
                    # implement stop-loss/take-profit and standard deviation or % drift
                else:
                    acceptable_price = 5000 # set to default of 5000

            print("Acceptable price : " + str(acceptable_price))

            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                #TODO fix the position size checks
                # check if the amount would go over the position limit, if yes, then fill up to position limit
                #if abs(-best_ask_amount + state.position[product]) > self.position_limit:
                #    best_ask_amount = self.position_limit - abs(state.position[product])
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))

            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                #TODO fix the position size checks
                # check if the amount would go over the position limit, if yes, then fill up to position limit
                #if abs(best_bid_amount + state.position[product]) > self.position_limit:
                #    best_bid_amount = self.position_limit - abs(state.position[product])
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))

            print("Own Orders: " + str(orders))

            result[product] = orders
            
            # if it is starfruit
            if product == "STARFRUIT":
                # calculate the mid_price
                price = (best_ask + best_bid) / 2
        
        # serialize the previous mid_prices for STARFRUIT and store those values
        traderData = self.serialize_state(state,price) # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.

        conversions = 1
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