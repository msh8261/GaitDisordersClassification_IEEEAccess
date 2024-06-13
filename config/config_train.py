# pylint: disable = c0411, c0301, c0103, c0303


params = dict(
    bfs_methods=[
        "ANOVA",
        "chi2",
        "kruskal",
        "cmim",
        "disr",
        "mifs",
        "ReliefF",
        "SURF",
        "SURFstar",
        "MultiSURF",
    ],
    # bfs_methods = ['ANOVA', 'chi2', "vit1D"],
    models_name=["gru", "lstm", "dnn"],
    # models_name = ["dnn"],
    sequences=300,
    num_class=3,
    # 0-6 -->> 0 means no augmentation, 5 is best
    num_augmentation=5,
    batch_size=64,  # 16  is best for gan (last padding), 64 is best gan (zero padding)
    epochs=100,  # 100  is best
    ## 'relu', 'sigmoid', 'tanh', 'selu','elu',
    ## 'LeakyReLU','rrelu'
    acf_indx="LeakyReLU",
    ## 'Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad',
    ## 'Adamax','Adamw', 'AdaBound'
    opt_indx="Adamw",
    last_layer=64,
    dropout=0.1,  # 0.1 best
    # set 10 for mlp and 300 for conv2d
    bottleneck=70,
    learning_rate=1e-2,  # 1e-2 best
    weight_decay=1e-5,  # 1e-5 best
    lr_decay=0.1,  # 0.1 best
    n_folds=5,
    # 2,5,11 are better and 2 is best among [1,2,3,4,5,11,21,42]
    random_state_list=[21],  # 21 best for gan zero padd
)
