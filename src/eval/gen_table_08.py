from main_eval import print_all_models_table

if __name__ == '__main__':
	print_all_models_table("data/mate/dev/predictions", accept_mention_correct=False, shots=[2], tasks=['i2i', 'd2d', 'i2d', 'd2i'])