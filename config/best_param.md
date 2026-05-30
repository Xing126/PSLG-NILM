# kettle（还需观察）
    segment_method: "fluss" # clasp / fluss / espresso
    # fluss specific params
    window_size: 25
    n_regimes: 3
    excl_factor: 5

# washing_machine
    segment_method: "fluss" # clasp / fluss / espresso / clasp-origin
    # fluss specific params
    window_size: 50
    n_regimes: 4
    excl_factor: 5