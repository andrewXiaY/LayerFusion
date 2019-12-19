OUTFEATURES = {"ssl_rotate": 4,
               "ssl_jigsaw": 24,
               "ssl_moveblur": 36,
               "ssl_exchange_pos": 9216*3,
               "ssl_noise_add": 4,
               "ssl_gaussian_blur": 10,
	       "ssl_box_blur": 10,
               "ssl_color": 9}	

DEFAULT_SHAPE = (96, 96)
BATCH_SIZE = 32
MAX_EPOCH = 20
TRAIN_DATAPATH = "./data/unsupervised_train"
TEST_DATAPATH = "./data/unsupervised_test"