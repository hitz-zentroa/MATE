from datasource.meta_correlation import utils as meta_correlation_utils

def get_dataset_utils(dataset_path: str):
	# Not super proud of this func but it'll do the job for now
	if "mate" in dataset_path:
		return meta_correlation_utils
	else:
		raise Exception(f"There are not dataset utils linked to the dataset '{dataset_path}'. Go to datasource_factory.py and link it.")