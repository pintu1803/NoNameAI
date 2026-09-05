
import config
import utils
import resnet18
import dataset

def main():
    model = resnet18.load_resnet18()
    data_set = dataset.load_data_from_cache()

if __name__ == "__main__":
    main()