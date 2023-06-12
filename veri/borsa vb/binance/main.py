from binance.client import Client
import keys

client=Client(api_key=keys.pk,api_secret=keys.sk)

balance=client.get_asset_balance("USDC")

#print(balance)

depth = client.get_order_book(symbol='BTCUSDT')

print(depth)