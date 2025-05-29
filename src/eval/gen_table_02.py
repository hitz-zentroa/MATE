from main_eval import print_attribute_table

if __name__ == '__main__':
	print_attribute_table("data/mate/dev/predictions/", accept_mention_correct=False, modalities=["mm"],
						  attribute_fncs=["target_attribute"])