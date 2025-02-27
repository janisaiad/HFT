publisher_id: uint16_t  # The publisher ID assigned by Databento, which denotes the dataset and venue.
instrument_id: uint32_t  # The numeric instrument ID.
ts_event: uint64_t  # The matching-engine-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
price: int64_t  # The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.
size: uint32_t  # The order quantity.
action: char  # The event action. Can be Add, Cancel, Modify, cleaR book, Trade, or Fill. See Action.
side: char  # The side that initiates the event. Can be Ask for the sell aggressor in a trade, Bid for the buy aggressor in a trade, or None where no side is specified by the original trade or the record was not a trade.
flags: uint8_t  # A bit field indicating event end, message characteristics, and data quality. See Flags.
depth: uint8_t  # The book level where the update event occurred.
ts_recv: uint64_t  # The capture-server-received timestamp expressed as the number of nanoseconds since the UNIX epoch.
ts_in_delta: int32_t  # The matching-engine-sending timestamp expressed as the number of nanoseconds before ts_recv.
sequence: uint32_t  # The message sequence number assigned at the venue.
bid_px_N: int64_t  # The bid price at level N (top level if N = 00).
ask_px_N: int64_t  # The ask price at level N (top level if N = 00).
bid_sz_N: uint32_t  # The bid size at level N (top level if N = 00).
ask_sz_N: uint32_t  # The ask size at level N (top level if N = 00).
bid_ct_N: uint32_t  # The number of bid orders at level N (top level if N = 00).
ask_ct_N: uint32_t  # The number of ask orders at level N (top level if N = 00)