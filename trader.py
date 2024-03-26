from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pickle


class Trader:


    def run(self, state: TradingState):
        # serialize the mid_price for Starfruit as part of it's trading strategy

        # Only method required. It takes all buy and sell orders for all symbols as an input,
        # and outputs a list of orders to be sent.
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            if product == "AMETHYSTS":
                acceptable_price = 10000
            elif product == "STARFRUIT":
                acceptable_price = 5000 - (state.timestamp * .025)  # Participant should calculate this value
            print(product)
            print()
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", " + str(order_depth.buy_orders) + ", Sell order depth : " + str(
                len(order_depth.sell_orders)) + ", " + str(order_depth.buy_orders))

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

        traderData = "SAMPLE"  # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
    
# Trading Strategies
def moving_avg(timeframe :int, state: TradingState) -> int:
        """
        Calculates the moving average based on the the past x days.
        - timeframe : past x days
        - state : current trading state

        returns the predicted price
        """