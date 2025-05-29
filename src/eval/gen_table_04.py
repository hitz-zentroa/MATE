from main_eval import print_cot_table

if __name__ == '__main__':
	print_cot_table("data/mate/dev/predictions", colums=["10"], add_ose=True, add_delta_to_cols=True)