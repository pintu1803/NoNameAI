
from matplotlib import pyplot as plt

def myLog(title):
    margin = "*" * ((78 - len(title))//2)
    print("\n")
    print("=" * 80)
    print(f"{margin} {title} {margin}")
    print("=" * 80)

def addLine():
    # print("\n")
    print("=" * 80)
    # print("\n")

def showImage(sample, label, time=2):
    plt.imshow(sample["image"])
    plt.title(label)

    plt.axis("off")
    plt.show(block=False) #don't pause code execution here, keep going on
    plt.pause(time)
    plt.close()