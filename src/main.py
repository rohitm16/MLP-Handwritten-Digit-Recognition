import os
import matplotlib.pyplot as plt
from src import mnist_loader
from src.network import Network

def main():
    # 1. Configuration
    DATA_PATH = 'data/mnist.pkl.gz'
    ASSETS_PATH = 'assets'
    
    if not os.path.exists(ASSETS_PATH):
        os.makedirs(ASSETS_PATH)

    # 2. Load Data
    print(f"Loading data from {DATA_PATH}...")
    try:
        training_data, validation_data, test_data = mnist_loader.load_data_wrapper(DATA_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Initialize Network
    print("Initializing Network [784, 30, 10]...")
    net = Network([784, 30, 10])

    # 4. Train
    EPOCHS = 30
    BATCH_SIZE = 10
    LEARNING_RATE = 3.0
    
    print(f"Training for {EPOCHS} epochs...")
    history = net.SGD(training_data, EPOCHS, BATCH_SIZE, LEARNING_RATE, test_data=test_data)

    # 5. Visualize Results
    print("Saving training history to assets/accuracy_plot.png...")
    plt.plot(history)
    plt.title('Network Accuracy over Epochs')
    plt.ylabel('Correct Test Examples')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.savefig(f'{ASSETS_PATH}/accuracy_plot.png')
    print("Done.")

if __name__ == "__main__":
    main()